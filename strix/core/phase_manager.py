"""Phase-based prompt transition state machine.

Instead of one massive prompt covering Discovery, Analysis, and Reporting
simultaneously, this module manages transitions between lean, focused prompts:

    State 1 — DISCOVERY:  Recon, scanning, enumeration (~500 tok prompt)
    State 2 — DEEP_DIVE:  Exploitation, validation, proof-of-concept (~500 tok)
    State 3 — REPORTING:  Final report compilation (~500 tok)

Transitions happen automatically based on findings and budget consumption,
or can be forced by the agent through a tool call.

Gated behind ``STRIX_PHASE_TRANSITIONS=true``.
"""

from __future__ import annotations

import enum
import json
import logging
from typing import Any


logger = logging.getLogger(__name__)


class ScanPhase(enum.Enum):
    """The three phases of a security scan lifecycle."""

    DISCOVERY = "discovery"
    DEEP_DIVE = "deep_dive"
    REPORTING = "reporting"


# Legal transitions — forward only: Discovery → Deep Dive → Reporting.
# Discovery → Reporting is also legal (skip deep dive for shallow scans).
_LEGAL_TRANSITIONS: dict[ScanPhase, frozenset[ScanPhase]] = {
    ScanPhase.DISCOVERY: frozenset({ScanPhase.DEEP_DIVE, ScanPhase.REPORTING}),
    ScanPhase.DEEP_DIVE: frozenset({ScanPhase.REPORTING}),
    ScanPhase.REPORTING: frozenset(),
}


# ---------------------------------------------------------------------------
# Phase-specific system prompt fragments (~500 tokens each)
# ---------------------------------------------------------------------------

_PHASE_PROMPTS: dict[ScanPhase, str] = {
    ScanPhase.DISCOVERY: """\
You are in the DISCOVERY phase. Your sole objective is reconnaissance and \
enumeration of the target.

Focus on:
- Identifying the attack surface: endpoints, services, ports, technologies.
- Mapping the application architecture and data flows.
- Enumerating authentication and authorization mechanisms.
- Discovering hidden paths, parameters, and APIs.
- Recording all findings in notes for the next phase.

Do NOT attempt exploitation yet. Gather intelligence methodically. When you \
have identified at least one promising attack vector, transition to the \
DEEP_DIVE phase.

Use `search_tools` to discover available scanning and enumeration tools.""",
    ScanPhase.DEEP_DIVE: """\
You are in the DEEP_DIVE phase. Your sole objective is to validate and \
exploit the attack vectors discovered during reconnaissance.

Focus on:
- Crafting and testing payloads against identified vectors.
- Validating vulnerabilities with proof-of-concept exploits.
- Escalating access where possible (privilege escalation, lateral movement).
- Documenting exact reproduction steps for each confirmed vulnerability.
- Recording severity assessments with evidence.

When you have validated all promising vectors or are approaching budget \
limits, transition to the REPORTING phase.

Use `search_tools` to find exploitation and analysis tools.""",
    ScanPhase.REPORTING: """\
You are in the REPORTING phase. Your sole objective is to compile and \
deliver the final penetration test report.

Focus on:
- Creating vulnerability reports for each confirmed finding.
- Including exact reproduction steps, payloads, and evidence.
- Assigning accurate severity ratings (CVSS where applicable).
- Writing an executive summary of the engagement.
- Calling finish_scan to deliver the report.

Do NOT start new lines of investigation. Wrap up and deliver.""",
}


class PhaseManager:
    """Manages scan phase transitions and phase-specific prompt selection.

    Serialisable to/from a dict for checkpoint persistence.
    """

    def __init__(self, initial_phase: ScanPhase = ScanPhase.DISCOVERY) -> None:
        self._phase = initial_phase
        self._transition_history: list[dict[str, str]] = []
        self._findings_count = 0

    @property
    def current_phase(self) -> ScanPhase:
        """The current scan phase."""
        return self._phase

    @property
    def phase_name(self) -> str:
        """Human-readable name of the current phase."""
        return self._phase.value

    @property
    def transition_history(self) -> list[dict[str, str]]:
        """List of past transitions for audit."""
        return list(self._transition_history)

    def record_finding(self) -> None:
        """Note that a finding was discovered (drives auto-transition heuristics)."""
        self._findings_count += 1

    def transition(self, to: ScanPhase, *, reason: str = "") -> bool:
        """Transition to a new phase if the move is legal.

        Returns ``True`` on success, ``False`` if the transition is invalid.
        """
        if to not in _LEGAL_TRANSITIONS.get(self._phase, frozenset()):
            logger.warning(
                "Illegal phase transition %s → %s (reason: %s)",
                self._phase.value,
                to.value,
                reason,
            )
            return False
        old = self._phase
        self._phase = to
        self._transition_history.append({
            "from": old.value,
            "to": to.value,
            "reason": reason,
        })
        logger.info("Phase transition: %s → %s (%s)", old.value, to.value, reason)
        return True

    def prompt_for_phase(self, phase: ScanPhase | None = None) -> str:
        """Return the phase-specific system prompt fragment."""
        return _PHASE_PROMPTS[phase or self._phase]

    def should_transition(
        self,
        *,
        budget_fraction: float = 0.0,
        findings_count: int | None = None,
    ) -> ScanPhase | None:
        """Heuristic: suggest a phase transition based on state.

        Returns the suggested phase or ``None`` if no transition is warranted.

        - DISCOVERY → DEEP_DIVE when findings are discovered.
        - DEEP_DIVE → REPORTING when budget is ≥ 70% consumed.
        - DISCOVERY → REPORTING when budget is ≥ 85% consumed (skip deep dive).
        """
        count = findings_count if findings_count is not None else self._findings_count
        if self._phase == ScanPhase.DISCOVERY:
            if budget_fraction >= 0.85:
                return ScanPhase.REPORTING
            if count > 0:
                return ScanPhase.DEEP_DIVE
        elif self._phase == ScanPhase.DEEP_DIVE:
            if budget_fraction >= 0.70:
                return ScanPhase.REPORTING
        return None

    # -- Serialisation for checkpoint persistence ----------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict for coordinator checkpointing."""
        return {
            "phase": self._phase.value,
            "findings_count": self._findings_count,
            "transition_history": list(self._transition_history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseManager:
        """Restore from a checkpoint dict."""
        phase_str = data.get("phase", ScanPhase.DISCOVERY.value)
        try:
            phase = ScanPhase(phase_str)
        except ValueError:
            logger.warning("Unknown phase %r in checkpoint; defaulting to DISCOVERY", phase_str)
            phase = ScanPhase.DISCOVERY
        mgr = cls(initial_phase=phase)
        mgr._findings_count = int(data.get("findings_count", 0))
        mgr._transition_history = list(data.get("transition_history", []))
        return mgr


def build_phase_system_prompt(
    *,
    base_prompt: str,
    phase_manager: PhaseManager,
    tool_index: str = "",
) -> str:
    """Compose a complete system prompt for the current phase.

    Structure:
        1. Phase-specific instructions (~500 tok)
        2. Tool index (~100 tok)
        3. Base scan context (targets, scope — variable)
    """
    parts = [phase_manager.prompt_for_phase()]
    if tool_index:
        parts.append(f"\n{tool_index}")
    if base_prompt:
        parts.append(f"\n{base_prompt}")
    return "\n".join(parts)
