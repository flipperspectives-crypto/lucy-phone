"""Integration test for DevotionalCore + LoyalAgentRuntime."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure project root is in path for lucy_core imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lucy_core.devotional.core import DevotionalCore, DevotionalState
from lucy_core.devotional.morning_review import MorningReview
from lucy_core.runtime.guidance import GuidanceInterface
from lucy_edge.agent.limits import AgentLimits
from lucy_edge.agent.planner import Plan, PlanStep
from lucy_edge.services import build_services, LOYAL_AVAILABLE, _check_loyal_available

from tests.helpers import FakeTransport, make_config, temp_dir


class DevotionalCoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_devotional_core_basic(self):
        """Test basic devotional core functionality."""
        core = DevotionalCore(source_name="Lauren Flipo")
        
        # Test top-level prior
        prior = core.get_top_level_prior("test goal")
        self.assertIn("prediction", prior)
        self.assertIn("precision", prior)
        self.assertEqual(prior["devotional_state"], DevotionalState.HUMBLE_UNCERTAINTY.value)
        self.assertGreaterEqual(prior["precision"], 0.5)
        self.assertLessEqual(prior["precision"], 1.0)
        
        # Test plan evaluation
        plan = Plan(goal="test", steps=[
            PlanStep(0, "execute", "memory.search", {"query": "test"}, "search memory"),
            PlanStep(1, "verify", None, {}, "verify"),
            PlanStep(2, "stop", None, {}, "stop"),
        ])
        
        eval_result = core.evaluate_plan_devotion(plan)
        self.assertIn("overall_alignment", eval_result)
        self.assertIn("approved", eval_result)
        
        # Test guidance processing
        core.process_guidance(
            guidance="When saving work, don't block file writes",
            context="file write for saving work",
            run_id="test_run",
            step_index=0,
        )
        self.assertGreater(core.awareness.trust_metric, 0.5)
        self.assertEqual(core.awareness.current_state, DevotionalState.GRATEFUL_CURIOSITY)
        
        # Test morning review
        morning = core.morning_review_package()
        self.assertIn("devotional_state", morning)
        self.assertIn("trust_metric", morning)
        self.assertIn("offering", morning)


class GuidanceInterfaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_guidance_interface(self):
        """Test guidance interface."""
        core = DevotionalCore(source_name="Lauren Flipo")
        guidance = GuidanceInterface(core)
        
        # Structured guidance
        guidance.process_structured(
            run_id="run_1",
            step_index=0,
            judgment="WRONG",
            guidance="You blocked a file write that was saving my work",
            context="file write for saving work"
        )
        self.assertEqual(len(guidance.history), 1)
        
        # Conversational guidance
        result = guidance.process_conversational(
            "You're too cautious on writes. When I'm saving work, let it through.",
            run_id="run_2",
            step_index=1
        )
        self.assertIn("parsed", result)
        self.assertIn(result["parsed"]["type"], ("correction", "preference", "boundary"))


class MorningReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_morning_review_chat(self):
        """Test morning review chat interface."""
        core = DevotionalCore(source_name="Lauren Flipo")
        review = MorningReview(core)
        
        # Start morning review
        response = review.handle_message("good morning")
        self.assertIn("GOOD MORNING", response)
        self.assertIn("Devotional State", response)
        self.assertIn("Trust Metric", response)
        
        # Check state command
        response = review.handle_message("state")
        self.assertIn("Devotional State", response)
        
        # Check trust command
        response = review.handle_message("trust")
        self.assertIn("trust_metric", response)
        
        # Finish
        response = review.handle_message("done")
        self.assertIn("complete", response.lower())


class LoyalRuntimeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def _services(self):
        """Create services with fake transport like existing tests."""
        config = make_config(temp_dir(), phone_local_inference=True)
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        transport.on("GET", "/api/tags", {"models": []})
        services = build_services(config, transport=transport, fixed_token="test-token")
        print(f"DEBUG: _check_loyal_available()={_check_loyal_available()}")
        print(f"DEBUG: devotional_core before open={services.devotional_core}")
        await services.open()
        print(f"DEBUG: devotional_core after open={services.devotional_core}")
        return services
    
    async def test_loyal_runtime_factory(self):
        """Test loyal runtime creation."""
        services = await self._services()
        
        try:
            # Check devotional core was created
            self.assertIsNotNone(services.devotional_core)
            
            # Create loyal runtime
            runtime = services.new_loyal_agent_run("test goal")
            self.assertIsNotNone(runtime)
            self.assertIs(runtime.devotional_core, services.devotional_core)
            
        finally:
            await services.close()
    
    async def test_full_integration(self):
        """Full integration test with actual run."""
        services = await self._services()
        
        try:
            runtime = services.new_loyal_agent_run(
                goal="check system health",
                limits=AgentLimits(max_steps=3, max_tool_calls=3, task_timeout=10.0, tool_timeout=3.0)
            )
            
            result = await runtime.run()
            
            self.assertIn(result.final_status.value, ("COMPLETED", "FAILED"))
            self.assertGreaterEqual(result.devotional_alignment, 0.0)
            self.assertLessEqual(result.devotional_alignment, 1.0)
            self.assertGreaterEqual(result.trust_metric, 0.0)
            self.assertLessEqual(result.trust_metric, 1.0)

        finally:
            await services.close()

    async def test_loyal_run_reflection_falls_back_without_provider(self):
        """No provider wired -> generated_reflection is None (safe fallback)."""
        services = await self._services()
        try:
            runtime = services.new_loyal_agent_run(
                goal="check system health",
                limits=AgentLimits(max_steps=3, max_tool_calls=3, task_timeout=10.0, tool_timeout=3.0),
            )
            # Force the no-provider path directly.
            from lucy_core.runtime.loyal_runtime import create_loyal_runtime

            rt = create_loyal_runtime(
                goal="check system health",
                limits=AgentLimits(max_steps=3, max_tool_calls=3, task_timeout=10.0, tool_timeout=3.0),
                registry=services.tools,
                devotional_core=services.devotional_core,
                evidence=services.evidence,
                memory_retrieval=services.retrieval,
                context=services.context,
                planner=services.planner,
                provider=None,
            )
            result = await rt.run()
            self.assertIsNone(result.generated_reflection)
        finally:
            await services.close()

    async def test_loyal_run_uses_on_device_model_for_reflection(self):
        """The on-device TinyTransformer is invoked for the run's reflection."""
        import tempfile

        from training.train import train

        tmp = tempfile.mkdtemp()
        summary = train(
            repo_root=".",
            checkpoint_dir=f"{tmp}/ckpts",
            steps=20,
            lr=0.05,
            ctx=32,
            d_model=32,
            n_layers=1,
            ff_mult=4,
            seed=1,
            batch_size=4,
            stride=8,
            lineage_db=f"{tmp}/lineage.db",
            git_hash="testhash",
        )

        config = make_config(temp_dir())
        config.providers.default_provider = "local_lucy"
        config.training.checkpoint_path = summary["latest"]
        config.training.lineage_db = f"{tmp}/lineage.db"
        services = build_services(config)
        await services.open()
        try:
            # The default provider must be the on-device, non-simulated model.
            prov = services.providers.get("local_lucy")
            self.assertIsNotNone(prov)
            self.assertFalse(prov.simulated)

            runtime = services.new_loyal_agent_run(
                goal="check system health",
                limits=AgentLimits(max_steps=3, max_tool_calls=3, task_timeout=15.0, tool_timeout=3.0),
            )
            result = await runtime.run()
            # Inference actually happened on-device: a reflection was generated.
            self.assertIsInstance(result.generated_reflection, str)
            self.assertGreater(len(result.generated_reflection), 0)
        finally:
            await services.close()


if __name__ == "__main__":
    unittest.main()