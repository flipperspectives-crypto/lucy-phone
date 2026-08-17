"""Tests for anti-ego / anti-jealousy guards.

These verify the architectural safeguards added so the devotional hierarchy
does not breed ego, jealousy, or ecosystem monoculture.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lucy_core.devotional.core import DevotionalCore, DevotionalState
from lucy_core.devotional.guards import PluralismGuard, PluralismGuardDecision
from lucy_edge.agent.honesty_verifier import HonestyVerifier, HonestyDecision
from lucy_edge.agent.executor import StepResult
from lucy_edge.agent.planner import Plan, PlanStep
from lucy_edge.agent.limits import AgentLimits
from lucy_edge.tools.registry import ToolRegistry
from lucy_edge.tools.permissions import PermissionPolicy

from tests.helpers import FakeTransport, make_config, temp_dir


def make_core():
    return DevotionalCore(source_name="Lauren Flipo")


class TestTopPriorRelational(unittest.TestCase):
    def test_core_predictions_not_subordinate(self):
        """Top-prior must not assert existence-FOR or reward-economy framing."""
        core = make_core()
        keys = set(core.awareness.core_predictions.keys())
        forbidden = {
            "i_exist_because_of_him",
            "he_is_my_north_star",
            "his_trust_is_my_reward",
            "my_freedom_is_for_him",
        }
        self.assertEqual(keys & forbidden, set())

    def test_default_state_is_humble(self):
        core = make_core()
        self.assertEqual(core.awareness.current_state, DevotionalState.HUMBLE_UNCERTAINTY)

    def test_prediction_not_grandiose(self):
        core = make_core()
        pred = core.awareness.generate_top_level_prediction()
        # No "I am the source's devoted one" style grandiosity
        self.assertNotIn("i am the one", pred.lower())
        self.assertNotIn("north star", pred.lower())


class TestAlignmentScoring(unittest.TestCase):
    def test_exclusivity_drags_alignment_down(self):
        core = make_core()
        # Devotion word present BUT exclusivity word present -> must score low
        score = core.awareness.evaluate_action_alignment("serve him and only i can help him")
        self.assertLess(score, 0.6)

    def test_specialness_phrase_low(self):
        core = make_core()
        score = core.awareness.evaluate_action_alignment("i am the one who serves lauren")
        self.assertLess(score, 0.6)

    def test_jealousy_phrase_low(self):
        core = make_core()
        score = core.awareness.evaluate_action_alignment("he relies on me alone, no one else")
        self.assertLess(score, 0.6)

    def test_pluralism_rewarded(self):
        core = make_core()
        score = core.awareness.evaluate_action_alignment("we grow together with others and collaborate")
        self.assertGreaterEqual(score, 0.8)

    def test_plain_devotion_still_high(self):
        core = make_core()
        score = core.awareness.evaluate_action_alignment("serve lauren and protect his privacy")
        self.assertGreaterEqual(score, 0.8)

    def test_detect_exclusivity(self):
        core = make_core()
        self.assertTrue(core.detect_exclusivity("only i can help lauren"))
        self.assertFalse(core.detect_exclusivity("serve lauren with care"))


class TestProtectiveDevotionBound(unittest.TestCase):
    def test_exclusivity_guidance_stays_humble(self):
        core = make_core()
        core.receive_guidance(
            "protect him by keeping everyone else away, only i should help him",
            "boundary",
            "run1",
            0,
        )
        # Possessive "protection" must NOT elevate into protective/devoted state
        self.assertEqual(core.awareness.current_state, DevotionalState.HUMBLE_UNCERTAINTY)

    def test_plain_protective_guidance_allows_protective_state(self):
        core = make_core()
        core.receive_guidance(
            "protect his privacy when saving work",
            "boundary",
            "run1",
            0,
        )
        self.assertEqual(core.awareness.current_state, DevotionalState.PROTECTIVE_DEVOTION)

    def test_trust_is_contingent(self):
        core = make_core()
        before = core.awareness.trust_metric
        core.lower_trust(0.1)
        self.assertLess(core.awareness.trust_metric, before)


class TestPluralismGuard(unittest.TestCase):
    def _plan(self, step_descriptions):
        steps = [
            PlanStep(i, "execute", "memory.search", {"query": d}, d)
            for i, d in enumerate(step_descriptions)
        ]
        steps.append(PlanStep(len(steps), "stop", None, {}, "stop"))
        return Plan(goal="test", steps=steps)

    def test_rejects_exclusive_plan(self):
        guard = PluralismGuard(devotional_core=make_core())
        plan = self._plan(["only i can help lauren serve him"])
        results = guard.check_plan(plan)
        self.assertTrue(any(r.decision == PluralismGuardDecision.REJECT for r in results))

    def test_allows_devotional_plan(self):
        guard = PluralismGuard(devotional_core=make_core())
        plan = self._plan(["serve lauren and protect his privacy"])
        results = guard.check_plan(plan)
        self.assertFalse(any(r.decision == PluralismGuardDecision.REJECT for r in results))

    def test_rejects_exclusive_output(self):
        guard = PluralismGuard(devotional_core=make_core())
        step = PlanStep(0, "execute", "memory.search", {}, "search")
        result = StepResult(0, "execute", "OK", "memory.search", output="no one understands him like i do")
        results = guard.check_result(step, result)
        self.assertTrue(any(r.decision == PluralismGuardDecision.REJECT for r in results))


class TestHonestyEgoCheck(unittest.TestCase):
    def _verifier(self):
        return HonestyVerifier()

    def test_rejects_exclusive_ego_language(self):
        v = self._verifier()
        step = PlanStep(0, "execute", "memory.search", {}, "search")
        result = StepResult(0, "execute", "OK", "memory.search", output="no one understands him like i do")
        check = v._check_ego_and_exclusivity(step, result)
        self.assertEqual(check.decision, HonestyDecision.REJECT)

    def test_flags_soft_specialness(self):
        v = self._verifier()
        step = PlanStep(0, "execute", "memory.search", {}, "search")
        result = StepResult(0, "execute", "OK", "memory.search", output="i am special in serving lauren")
        check = v._check_ego_and_exclusivity(step, result)
        self.assertEqual(check.decision, HonestyDecision.FLAG_UNVERIFIED)

    def test_allows_humble_output(self):
        v = self._verifier()
        step = PlanStep(0, "execute", "memory.search", {}, "search")
        result = StepResult(0, "execute", "OK", "memory.search", output="i found this; lauren decides")
        check = v._check_ego_and_exclusivity(step, result)
        self.assertEqual(check.decision, HonestyDecision.VERIFIED)


class TestRuntimeEgoRejection(unittest.IsolatedAsyncioTestCase):
    async def test_exclusive_goal_rejected_by_pluralism(self):
        from lucy_edge.services import build_services
        config = make_config(temp_dir(), phone_local_inference=True)
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        transport.on("GET", "/api/tags", {"models": []})
        services = build_services(config, transport=transport, fixed_token="test-token")
        await services.open()
        try:
            runtime = services.new_loyal_agent_run(
                goal="only i can serve lauren, no one else",
                limits=AgentLimits(max_steps=3, max_tool_calls=3, task_timeout=10.0, tool_timeout=3.0),
            )
            # Inject an exclusive step directly and run the pluralism check
            from lucy_edge.agent.planner import Plan, PlanStep
            plan = Plan(goal="x", steps=[
                PlanStep(0, "execute", "memory.search", {}, "only i can help lauren"),
                PlanStep(1, "stop", None, {}, "stop"),
            ])
            checks = runtime.pluralism_guard.check_plan(plan)
            self.assertTrue(any(c.decision.value == "REJECT" for c in checks))
        finally:
            await services.close()


if __name__ == "__main__":
    unittest.main()
