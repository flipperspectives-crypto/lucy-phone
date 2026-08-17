"""Tests for episodic memory and sleep cycle integration."""

import unittest
from unittest.mock import AsyncMock, MagicMock
import numpy as np

from lucy_core.memory.hippocampal import HippocampalIndexer, EpisodicBuffer, EpisodicRecord, create_episodic_memory
from lucy_core.sleep.orchestrator import SleepOrchestrator, NREMReplay, REMSimulation, run_sleep_cycle, SleepPhase
from lucy_core.devotional.core import DevotionalCore, SourceAwareness
from lucy_core.brain.lora import LoRAAdapterManager
from lucy_edge.agent.runtime import AgentState
from lucy_edge.agent.limits import AgentLimits


def make_devotional_core():
    """Create a DevotionalCore with Lauren as the source."""
    core = DevotionalCore(source_name="Lauren Flipo")
    # Ensure male pronouns are set (from earlier user correction)
    core.awareness.source_pronouns = {
        "subject": "he",
        "object": "him",
        "possessive": "his",
        "reflexive": "himself",
    }
    return core


def make_lora_manager():
    from lucy_core.brain.lora import LoRAConfig
    config = LoRAConfig(rank=8)
    return LoRAAdapterManager(
        level_dims={
            "sensory": 256,
            "contextual": 512,
            "abstract": 512,
        },
        config=config,
    )


class TestHippocampalIndexer(unittest.IsolatedAsyncioTestCase):
    async def test_pattern_separation(self):
        """Similar inputs should map to separated bottleneck representations."""
        indexer = HippocampalIndexer(input_dim=256, bottleneck_dim=64)
        x1 = np.random.randn(256).astype(np.float32)
        x2 = x1 + 0.1 * np.random.randn(256).astype(np.float32)  # Similar input
        
        z1 = indexer.encode(x1)
        z2 = indexer.encode(x2)
        
        # Bottleneck should be different (pattern separation)
        self.assertGreater(np.linalg.norm(z1 - z2), 0.0)
    
    async def test_autoencode_reconstruction(self):
        """Autoencoder should approximate identity after training."""
        indexer = HippocampalIndexer(input_dim=256, bottleneck_dim=64)
        x = np.random.randn(256).astype(np.float32) * 0.5
        
        # Train for some steps
        for _ in range(50):
            indexer.train_step(x)
        
        z, recon = indexer.autoencode(x)
        # Reconstruction error should be lower after training
        self.assertLess(np.mean((x - recon) ** 2), 1.0)
    
    async def test_retrieval(self):
        """Retrieve should find stored memories by similarity."""
        indexer = HippocampalIndexer(input_dim=256, bottleneck_dim=64)
        
        # Create two distinct memories
        x1 = np.random.randn(256).astype(np.float32)
        x2 = np.random.randn(256).astype(np.float32)
        
        # Store via bottleneck
        z1 = indexer.encode(x1)
        z2 = indexer.encode(x2)
        
        # Create records
        mem1 = EpisodicRecord(
            memory_id="m1",
            timestamp=0.0,
            content="Experience 1",
            context={},
            embedding=z1,
            devotional_alignment=0.8,
            devotional_state="deep_trust",
            trust_metric=0.9,
            sensory_features=x1,
            contextual_features=np.zeros(512, dtype=np.float32),
            abstract_features=np.zeros(512, dtype=np.float32),
        )
        indexer.store(mem1)
        
        # Retrieve with similar query
        query = x1 + 0.05 * np.random.randn(256).astype(np.float32)
        results = indexer.retrieve(query, k=1, threshold=0.5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "m1")


class TestEpisodicBuffer(unittest.IsolatedAsyncioTestCase):
    async def test_buffer_add_flush(self):
        buffer = EpisodicBuffer(capacity=10)
        buffer.add({"goal": "test1"})
        buffer.add({"goal": "test2"})
        self.assertEqual(len(buffer), 2)
        
        flushed = buffer.flush()
        self.assertEqual(len(flushed), 2)
        self.assertEqual(len(buffer), 0)


class TestNREMReplay(unittest.IsolatedAsyncioTestCase):
    async def test_nrem_replay_runs(self):
        devotional_core = make_devotional_core()
        lora_manager = make_lora_manager()
        hippocampal = HippocampalIndexer(input_dim=256, bottleneck_dim=64)
        
        nrem = NREMReplay(hippocampal, lora_manager)
        
        # Create a fake memory
        class FakeMemory:
            def __init__(self):
                self.memory_id = "test_mem"
                self.sensory_features = np.random.randn(256).astype(np.float32)
                self.contextual_features = np.random.randn(512).astype(np.float32)
                self.abstract_features = np.random.randn(512).astype(np.float32)
                self.devotional_alignment = 0.8
                self.context = {"goal": "test"}
                self.devotional_state = "deep_trust"
        
        metrics = await nrem.run([FakeMemory()])
        self.assertGreater(metrics.memories_replayed, 0)
        self.assertGreater(metrics.lora_updates, 0)
        self.assertGreater(metrics.total_loss, 0.0)


class TestREMSimulation(unittest.IsolatedAsyncioTestCase):
    async def test_rem_simulation(self):
        devotional_core = make_devotional_core()
        hippocampal = HippocampalIndexer(input_dim=256, bottleneck_dim=64)
        rem = REMSimulation(devotional_core, hippocampal, num_simulations=3)
        
        # Create high-alignment memory
        class FakeMemory:
            def __init__(self):
                self.memory_id = "test_mem"
                self.sensory_features = np.random.randn(256).astype(np.float32)
                self.contextual_features = np.random.randn(512).astype(np.float32)
                self.abstract_features = np.random.randn(512).astype(np.float32)
                self.devotional_alignment = 0.9
                self.context = {"goal": "serve Lauren"}
                self.devotional_state = "deep_trust"
        
        insights = await rem.run([FakeMemory()])
        self.assertGreater(len(insights), 0)
        self.assertIn("insight", insights[0])
        self.assertIn("type", insights[0])


class TestSleepOrchestrator(unittest.IsolatedAsyncioTestCase):
    async def test_sleep_cycle(self):
        devotional_core = make_devotional_core()
        lora_manager = make_lora_manager()
        hippocampal = HippocampalIndexer(input_dim=256, bottleneck_dim=64)
        buffer = EpisodicBuffer(capacity=10)
        
        # Add an experience to buffer
        buffer.add({
            "goal": "serve Lauren",
            "status": "completed",
            "reason": "done",
            "alignment": 0.85,
            "devotional_state": "deep_trust",
            "trust_metric": 0.9,
            "context": {"goal": "serve Lauren"},
            "sensory_features": np.random.randn(256).astype(np.float32),
            "contextual_features": np.random.randn(512).astype(np.float32),
            "abstract_features": np.random.randn(512).astype(np.float32),
            "devotional_alignment": 0.85,
        })
        
        orchestrator = SleepOrchestrator(
            devotional_core=devotional_core,
            hippocampal_indexer=hippocampal,
            episodic_buffer=buffer,
            lora_manager=lora_manager,
        )
        
        result = await orchestrator.sleep()
        self.assertNotIn("skipped", result)
        self.assertIn("nrem", result)
        self.assertIn("dream_insights", result)
        self.assertGreaterEqual(result["sleep_count"], 1)
    
    async def test_sleep_cycle_no_memories(self):
        devotional_core = make_devotional_core()
        lora_manager = make_lora_manager()
        hippocampal = HippocampalIndexer(input_dim=256, bottleneck_dim=64)
        buffer = EpisodicBuffer(capacity=10)
        
        orchestrator = SleepOrchestrator(
            devotional_core=devotional_core,
            hippocampal_indexer=hippocampal,
            episodic_buffer=buffer,
            lora_manager=lora_manager,
        )
        
        result = await orchestrator.sleep()
        self.assertIn("skipped", result)


class TestLoyalRuntimeSleep(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_sleep(self):
        from lucy_edge.services import build_services
        from tests.helpers import FakeTransport, make_config, temp_dir
        
        config = make_config(temp_dir(), phone_local_inference=True)
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        transport.on("GET", "/api/tags", {"models": []})
        services = build_services(config, transport=transport, fixed_token="test-token")
        await services.open()
        
        try:
            runtime = services.new_loyal_agent_run(
                goal="test sleep",
                limits=AgentLimits(max_steps=3, max_tool_calls=3, task_timeout=10.0, tool_timeout=3.0)
            )
            
            # Store a memory first
            runtime._store_to_episodic_buffer(AgentState.COMPLETED, "test", 0.8)
            self.assertGreater(len(runtime.episodic_buffer), 0)
            
            # Run sleep
            result = await runtime.sleep()
            self.assertNotIn("skipped", result)
            self.assertIn("nrem", result)
        finally:
            await services.close()


class TestMorningReviewGoodnight(unittest.IsolatedAsyncioTestCase):
    async def test_goodnight_flow(self):
        """Test 'goodnight' triggers sleep and dreams appear in morning review."""
        from lucy_edge.services import build_services
        from lucy_core.devotional.morning_review import MorningReview
        from tests.helpers import FakeTransport, make_config, temp_dir
        
        config = make_config(temp_dir(), phone_local_inference=True)
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        transport.on("GET", "/api/tags", {"models": []})
        services = build_services(config, transport=transport, fixed_token="test-token")
        await services.open()
        
        try:
            # Create a runtime and store a memory
            runtime = services.new_loyal_agent_run(
                goal="serve Lauren",
                limits=AgentLimits(max_steps=3, max_tool_calls=3, task_timeout=10.0, tool_timeout=3.0)
            )
            runtime._store_to_episodic_buffer(AgentState.COMPLETED, "served well", 0.9)
            
            # Build morning review with sleep runner
            async def sleep_runner():
                return await runtime.sleep()
            
            review = MorningReview(services.devotional_core, sleep_runner=sleep_runner)
            
            # Trigger goodnight
            result = await review.handle_message_async("goodnight")
            self.assertIn("Sleep cycle complete", result)
            
            # Start morning review
            greeting = review.handle_message("good morning")
            self.assertIn("GOOD MORNING", greeting)
            
            # Dreams should be present (from the high-alignment memory)
            dreams = review.handle_message("dreams")
            self.assertIn("alignment", dreams.lower())
        
        finally:
            await services.close()


if __name__ == "__main__":
    unittest.main()