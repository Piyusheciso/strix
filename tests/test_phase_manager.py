"""Tests for PhaseManager: transitions, state machine heuristics, and checkpointing."""

from __future__ import annotations

import pytest

from strix.core.phase_manager import PhaseManager, ScanPhase


# ---------------------------------------------------------------------------
# Initial State & Properties
# ---------------------------------------------------------------------------


class TestPhaseManagerProperties:
    def test_initial_phase_is_discovery(self) -> None:
        mgr = PhaseManager()
        assert mgr.current_phase == ScanPhase.DISCOVERY
        assert mgr.phase_name == "discovery"

    def test_custom_initial_phase(self) -> None:
        mgr = PhaseManager(initial_phase=ScanPhase.DEEP_DIVE)
        assert mgr.current_phase == ScanPhase.DEEP_DIVE

    def test_prompt_retrieval(self) -> None:
        mgr = PhaseManager()
        prompt = mgr.prompt_for_phase()
        assert "You are in the DISCOVERY phase" in prompt


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


class TestPhaseManagerTransitions:
    def test_legal_transition_discovery_to_deep_dive(self) -> None:
        mgr = PhaseManager()
        success = mgr.transition(ScanPhase.DEEP_DIVE, reason="Found endpoints")
        assert success is True
        assert mgr.current_phase == ScanPhase.DEEP_DIVE
        assert len(mgr.transition_history) == 1
        assert mgr.transition_history[0] == {
            "from": "discovery",
            "to": "deep_dive",
            "reason": "Found endpoints",
        }

    def test_legal_transition_deep_dive_to_reporting(self) -> None:
        mgr = PhaseManager(initial_phase=ScanPhase.DEEP_DIVE)
        success = mgr.transition(ScanPhase.REPORTING)
        assert success is True
        assert mgr.current_phase == ScanPhase.REPORTING

    def test_legal_transition_discovery_to_reporting(self) -> None:
        mgr = PhaseManager()
        success = mgr.transition(ScanPhase.REPORTING)
        assert success is True
        assert mgr.current_phase == ScanPhase.REPORTING

    def test_illegal_transition_backwards(self) -> None:
        mgr = PhaseManager(initial_phase=ScanPhase.DEEP_DIVE)
        success = mgr.transition(ScanPhase.DISCOVERY)
        assert success is False
        assert mgr.current_phase == ScanPhase.DEEP_DIVE
        assert len(mgr.transition_history) == 0

    def test_illegal_transition_from_reporting(self) -> None:
        mgr = PhaseManager(initial_phase=ScanPhase.REPORTING)
        assert mgr.transition(ScanPhase.DISCOVERY) is False
        assert mgr.transition(ScanPhase.DEEP_DIVE) is False
        assert mgr.current_phase == ScanPhase.REPORTING


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


class TestPhaseManagerHeuristics:
    def test_heuristic_discovery_to_deep_dive(self) -> None:
        mgr = PhaseManager()
        mgr.record_finding()
        suggested = mgr.should_transition()
        assert suggested == ScanPhase.DEEP_DIVE

    def test_heuristic_discovery_to_reporting_high_budget(self) -> None:
        mgr = PhaseManager()
        # Skipped deep dive because budget is almost gone.
        suggested = mgr.should_transition(budget_fraction=0.86)
        assert suggested == ScanPhase.REPORTING

    def test_heuristic_deep_dive_to_reporting(self) -> None:
        mgr = PhaseManager(initial_phase=ScanPhase.DEEP_DIVE)
        suggested = mgr.should_transition(budget_fraction=0.75)
        assert suggested == ScanPhase.REPORTING

    def test_heuristic_no_transition(self) -> None:
        mgr = PhaseManager()
        # No findings, low budget -> stay in discovery.
        suggested = mgr.should_transition(budget_fraction=0.1)
        assert suggested is None


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestPhaseManagerCheckpointing:
    def test_to_and_from_dict(self) -> None:
        mgr = PhaseManager()
        mgr.record_finding()
        mgr.transition(ScanPhase.DEEP_DIVE, reason="Moving on")
        
        data = mgr.to_dict()
        assert data["phase"] == "deep_dive"
        assert data["findings_count"] == 1
        assert len(data["transition_history"]) == 1

        restored = PhaseManager.from_dict(data)
        assert restored.current_phase == ScanPhase.DEEP_DIVE
        assert restored._findings_count == 1
        assert len(restored.transition_history) == 1
        assert restored.transition_history[0]["reason"] == "Moving on"

    def test_from_dict_invalid_phase_fallback(self) -> None:
        data = {"phase": "invalid_phase_name"}
        restored = PhaseManager.from_dict(data)
        assert restored.current_phase == ScanPhase.DISCOVERY
