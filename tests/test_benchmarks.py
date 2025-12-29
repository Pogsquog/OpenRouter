"""
Performance benchmarks for the OpenRouter library.
"""

import asyncio
import time
import pytest
from typing import List, Dict, Any
from openrouter import AsyncOpenRouter


class BenchmarkRunner:
    """Helper class to run performance benchmarks."""
    
    def __init__(self, client: AsyncOpenRouter):
        self.client = client
        self.results = {}
    
    async def benchmark_method(self, method_name: str, method_func, *args, **kwargs) -> Dict[str, Any]:
        """Run a benchmark for a specific method."""
        times = []
        iterations = kwargs.pop('iterations', 5)  # Default to 5 iterations
        
        for _ in range(iterations):
            start_time = time.time()
            try:
                await method_func(*args, **kwargs)
            except Exception:
                # If the method fails, record a high time to indicate failure
                times.append(float('inf'))
                continue
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times) if times else float('inf')
        min_time = min(times) if times else float('inf')
        max_time = max(times) if times else float('inf')
        
        result = {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'iterations': iterations,
            'times': times
        }
        
        self.results[method_name] = result
        return result


@pytest.mark.asyncio
async def test_performance_benchmarks():
    """Run performance benchmarks for critical operations."""
    import os
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set, skipping performance test")
    
    client = AsyncOpenRouter(api_key=api_key)
    benchmark = BenchmarkRunner(client)
    
    try:
        # Benchmark listing models
        models_benchmark = await benchmark.benchmark_method(
            'list_models', 
            client.list_models,
            iterations=3
        )
        print(f"List models - Avg: {models_benchmark['avg_time']:.3f}s, Min: {models_benchmark['min_time']:.3f}s, Max: {models_benchmark['max_time']:.3f}s")
        
        # Benchmark getting model info (only if we have models)
        models = await client.list_models()
        if models:
            first_model_id = models[0].id
            model_info_benchmark = await benchmark.benchmark_method(
                'get_model_info',
                client.get_model_info,
                first_model_id,
                iterations=3
            )
            print(f"Get model info - Avg: {model_info_benchmark['avg_time']:.3f}s, Min: {model_info_benchmark['min_time']:.3f}s, Max: {model_info_benchmark['max_time']:.3f}s")
        
        # Benchmark getting rate limits
        rate_limits_benchmark = await benchmark.benchmark_method(
            'get_rate_limits',
            client.get_rate_limits,
            iterations=3
        )
        print(f"Get rate limits - Avg: {rate_limits_benchmark['avg_time']:.3f}s, Min: {rate_limits_benchmark['min_time']:.3f}s, Max: {rate_limits_benchmark['max_time']:.3f}s")
        
        # Benchmark cost calculation (this is a local operation, should be very fast)
        cost_benchmark = await benchmark.benchmark_method(
            'calculate_cost',
            lambda: client.calculate_cost("openai/gpt-3.5-turbo", {"prompt_tokens": 100, "completion_tokens": 50}),
            iterations=100
        )
        print(f"Calculate cost - Avg: {cost_benchmark['avg_time']:.6f}s, Min: {cost_benchmark['min_time']:.6f}s, Max: {cost_benchmark['max_time']:.6f}s")
        
        # Performance expectations - all operations should complete in reasonable time
        assert models_benchmark['avg_time'] < 10.0, f"List models took too long: {models_benchmark['avg_time']:.3f}s"
        assert rate_limits_benchmark['avg_time'] < 10.0, f"Get rate limits took too long: {rate_limits_benchmark['avg_time']:.3f}s"
        assert cost_benchmark['avg_time'] < 0.1, f"Calculate cost took too long: {cost_benchmark['avg_time']:.6f}s"
        
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_concurrent_performance():
    """Test performance under concurrent load."""
    import os
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set, skipping performance test")
    
    client = AsyncOpenRouter(api_key=api_key)
    
    try:
        # Run multiple list_models requests concurrently
        start_time = time.time()
        tasks = [client.list_models() for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Filter out any exceptions
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        print(f"Concurrent list_models (3 requests) - Total time: {total_time:.3f}s, Successful: {len(successful_results)}/3")
        
        # Should complete in reasonable time
        assert total_time < 20.0, f"Concurrent requests took too long: {total_time:.3f}s"
        assert len(successful_results) > 0, "No concurrent requests succeeded"
        
    finally:
        await client.close()


def test_sync_performance():
    """Test performance of sync wrapper methods."""
    import os
    from openrouter import OpenRouter
    import time
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set, skipping performance test")
    
    client = OpenRouter(api_key=api_key)
    
    try:
        # Benchmark sync list_models
        start_time = time.time()
        models = client.list_models()
        end_time = time.time()
        
        sync_time = end_time - start_time
        print(f"Sync list_models - Time: {sync_time:.3f}s")
        
        assert sync_time < 10.0, f"Sync list models took too long: {sync_time:.3f}s"
        
        if models:
            # Benchmark sync get_model_info
            start_time = time.time()
            model_info = client.get_model_info(models[0].id)
            end_time = time.time()
            
            sync_model_time = end_time - start_time
            print(f"Sync get_model_info - Time: {sync_model_time:.3f}s")
            
            assert sync_model_time < 10.0, f"Sync get model info took too long: {sync_model_time:.3f}s"
        
    finally:
        client.close()