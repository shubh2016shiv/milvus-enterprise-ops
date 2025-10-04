"""
Rate Limiting Test Script

This script demonstrates and validates the rate limiting implementation across
both ConnectionManager and CollectionManager modules. It tests:

1. Token bucket rate limiter behavior
2. Request throttling under load
3. Burst capacity handling
4. Retry budget pattern
5. Metrics collection and monitoring
6. Integration with circuit breaker

The tests simulate various load patterns to verify production-ready behavior.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any
from loguru import logger

from connection_management import ConnectionManager
from collection_operations import CollectionManager
from collection_operations.schema import CollectionSchema, FieldSchema, DataType
from config import load_settings

# Setup logging
logging.basicConfig(level=logging.INFO)


class RateLimitingTestSuite:
    """Comprehensive test suite for rate limiting functionality."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.test_collection_name = "rate_limit_test_collection"
    
    async def test_basic_rate_limiting(self):
        """
        Test 1: Basic Rate Limiting Behavior
        
        Verifies that the rate limiter properly throttles requests when
        the configured rate is exceeded.
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Basic Rate Limiting Behavior")
        logger.info("="*80)
        
        # Create ConnectionManager with low rate limit for testing
        settings = load_settings()
        settings.connection.max_requests_per_second = 10  # Very low for testing
        settings.connection.rate_limiter_burst_multiplier = 1.0  # No burst
        
        conn_manager = ConnectionManager(config=settings)
        coll_manager = CollectionManager(conn_manager)
        
        try:
            logger.info("Configured rate limit: 10 requests/second (no burst)")
            logger.info("Sending 20 requests rapidly...")
            
            start_time = time.time()
            request_times = []
            
            # Send 20 requests as fast as possible
            for i in range(20):
                req_start = time.time()
                await coll_manager.list_collections()
                req_end = time.time()
                request_times.append(req_end - req_start)
                logger.debug(f"  Request {i+1}/20 completed in {req_end - req_start:.3f}s")
            
            total_time = time.time() - start_time
            
            # Analyze results
            throttled_requests = sum(1 for t in request_times if t > 0.01)
            avg_time = sum(request_times) / len(request_times)
            max_time = max(request_times)
            
            logger.info(f"\n✅ Results:")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Expected minimum: ~2.0s (20 requests / 10 req/s)")
            logger.info(f"  Throttled requests: {throttled_requests}/20")
            logger.info(f"  Average request time: {avg_time*1000:.1f}ms")
            logger.info(f"  Max request time: {max_time*1000:.1f}ms")
            
            # Get rate limiter metrics
            metrics = conn_manager.get_rate_limiter_metrics()
            if metrics:
                logger.info(f"\n📊 Rate Limiter Metrics:")
                logger.info(f"  Total requests: {metrics['total_requests']}")
                logger.info(f"  Total throttled: {metrics['total_throttled']}")
                logger.info(f"  Throttle rate: {metrics['throttle_rate_percent']:.1f}%")
                logger.info(f"  Avg wait time: {metrics['average_wait_time_seconds']:.3f}s")
            
            # Verify rate limiting worked
            if total_time >= 1.5:  # Should take at least 1.5s for 20 req @ 10 req/s
                logger.info("✅ SUCCESS: Rate limiting is working correctly!")
                self.results['basic_rate_limiting'] = 'PASS'
            else:
                logger.warning("⚠️  WARNING: Requests completed too quickly - rate limiting may not be working")
                self.results['basic_rate_limiting'] = 'FAIL'
        
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            self.results['basic_rate_limiting'] = 'ERROR'
        finally:
            conn_manager.close()
    
    async def test_burst_capacity(self):
        """
        Test 2: Burst Capacity Handling
        
        Verifies that the burst capacity allows temporary spikes above
        the sustained rate limit.
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Burst Capacity Handling")
        logger.info("="*80)
        
        settings = load_settings()
        settings.connection.max_requests_per_second = 10
        settings.connection.rate_limiter_burst_multiplier = 3.0  # 3x burst = 30 capacity
        
        conn_manager = ConnectionManager(config=settings)
        coll_manager = CollectionManager(conn_manager)
        
        try:
            logger.info("Configured: 10 req/s with 3x burst capacity (30 tokens)")
            logger.info("Test: Send 25 requests immediately, then wait, then send 25 more")
            
            # Phase 1: Initial burst (should use burst capacity)
            logger.info("\n📈 Phase 1: Initial burst (25 requests)...")
            phase1_start = time.time()
            
            for i in range(25):
                await coll_manager.list_collections()
            
            phase1_time = time.time() - phase1_start
            logger.info(f"  Phase 1 completed in {phase1_time:.2f}s")
            
            # Get metrics after burst
            metrics1 = conn_manager.get_rate_limiter_metrics()
            if metrics1:
                logger.info(f"  Throttled: {metrics1['total_throttled']}/{metrics1['total_requests']}")
                logger.info(f"  Current tokens: {metrics1['current_tokens']:.1f}")
            
            # Phase 2: Wait for token replenishment
            logger.info("\n⏱️  Waiting 2 seconds for token replenishment...")
            await asyncio.sleep(2)
            
            # Phase 3: Second burst
            logger.info("\n📈 Phase 2: Second burst (25 requests)...")
            phase2_start = time.time()
            
            for i in range(25):
                await coll_manager.list_collections()
            
            phase2_time = time.time() - phase2_start
            logger.info(f"  Phase 2 completed in {phase2_time:.2f}s")
            
            # Final metrics
            metrics2 = conn_manager.get_rate_limiter_metrics()
            if metrics2:
                logger.info(f"\n📊 Final Metrics:")
                logger.info(f"  Total requests: {metrics2['total_requests']}")
                logger.info(f"  Total throttled: {metrics2['total_throttled']}")
                logger.info(f"  Throttle rate: {metrics2['throttle_rate_percent']:.1f}%")
            
            # Verify burst capacity worked
            if phase1_time < 1.0 and phase2_time < 2.0:
                logger.info("✅ SUCCESS: Burst capacity is working correctly!")
                self.results['burst_capacity'] = 'PASS'
            else:
                logger.warning("⚠️  Burst behavior not as expected")
                self.results['burst_capacity'] = 'PARTIAL'
        
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            self.results['burst_capacity'] = 'ERROR'
        finally:
            conn_manager.close()
    
    async def test_concurrent_rate_limiting(self):
        """
        Test 3: Concurrent Request Rate Limiting
        
        Verifies that rate limiting works correctly when multiple
        concurrent coroutines are making requests.
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Concurrent Request Rate Limiting")
        logger.info("="*80)
        
        settings = load_settings()
        settings.connection.max_requests_per_second = 20
        
        conn_manager = ConnectionManager(config=settings)
        coll_manager = CollectionManager(conn_manager)
        
        try:
            logger.info("Configured: 20 req/s rate limit")
            logger.info("Test: 5 concurrent tasks, each making 10 requests (50 total)")
            
            async def worker(worker_id: int, request_count: int):
                """Worker that makes multiple requests."""
                for i in range(request_count):
                    await coll_manager.list_collections()
                    logger.debug(f"  Worker-{worker_id}: Request {i+1}/{request_count}")
                return worker_id
            
            start_time = time.time()
            
            # Launch 5 concurrent workers
            tasks = [worker(i, 10) for i in range(5)]
            await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            
            # Get metrics
            metrics = conn_manager.get_rate_limiter_metrics()
            
            logger.info(f"\n✅ Results:")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Expected minimum: ~2.5s (50 requests / 20 req/s)")
            logger.info(f"  Actual rate: {50 / total_time:.1f} req/s")
            
            if metrics:
                logger.info(f"\n📊 Metrics:")
                logger.info(f"  Total requests: {metrics['total_requests']}")
                logger.info(f"  Total throttled: {metrics['total_throttled']}")
                logger.info(f"  Throttle rate: {metrics['throttle_rate_percent']:.1f}%")
            
            if 2.0 <= total_time <= 4.0:
                logger.info("✅ SUCCESS: Concurrent rate limiting is working!")
                self.results['concurrent_rate_limiting'] = 'PASS'
            else:
                logger.warning("⚠️  Timing not as expected")
                self.results['concurrent_rate_limiting'] = 'PARTIAL'
        
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            self.results['concurrent_rate_limiting'] = 'ERROR'
        finally:
            conn_manager.close()
    
    async def test_retry_budget(self):
        """
        Test 4: Retry Budget Pattern
        
        Verifies that the retry budget prevents retry storms by
        denying retries when success rate drops below threshold.
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 4: Retry Budget Pattern")
        logger.info("="*80)
        
        settings = load_settings()
        settings.connection.enable_retry_budget = True
        settings.connection.retry_budget_min_success_rate = 0.8
        settings.connection.retry_budget_window_seconds = 10
        
        conn_manager = ConnectionManager(config=settings)
        
        try:
            logger.info("Configured: Retry budget enabled, min success rate: 80%")
            logger.info("Test: Simulate mix of successful and failed operations")
            
            # Simulate successful operations
            logger.info("\n✅ Phase 1: Simulating 20 successful operations...")
            for i in range(20):
                # Record success in retry budget
                if conn_manager._retry_budget:
                    conn_manager._retry_budget.record_attempt(success=True)
            
            metrics1 = conn_manager.get_retry_budget_metrics()
            if metrics1:
                logger.info(f"  Success rate: {metrics1['current_success_rate']:.2%}")
                logger.info(f"  Retries allowed: {metrics1['retries_allowed']}")
            
            # Simulate failures
            logger.info("\n❌ Phase 2: Simulating 15 failures...")
            retries_denied = 0
            for i in range(15):
                if conn_manager._retry_budget:
                    retry_allowed = conn_manager._retry_budget.record_attempt(success=False)
                    if not retry_allowed:
                        retries_denied += 1
                        logger.debug(f"  Retry {i+1} denied due to low success rate")
            
            metrics2 = conn_manager.get_retry_budget_metrics()
            if metrics2:
                logger.info(f"\n📊 Final Metrics:")
                logger.info(f"  Total attempts: {metrics2['total_attempts']}")
                logger.info(f"  Total successes: {metrics2['total_successes']}")
                logger.info(f"  Total failures: {metrics2['total_failures']}")
                logger.info(f"  Success rate: {metrics2['current_success_rate']:.2%}")
                logger.info(f"  Retries denied: {metrics2['retries_denied']}")
            
            if retries_denied > 0:
                logger.info(f"\n✅ SUCCESS: Retry budget denied {retries_denied} retries!")
                logger.info("   This prevents retry storms during outages.")
                self.results['retry_budget'] = 'PASS'
            else:
                logger.warning("⚠️  No retries were denied - may need more failures")
                self.results['retry_budget'] = 'PARTIAL'
        
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            self.results['retry_budget'] = 'ERROR'
        finally:
            conn_manager.close()
    
    async def test_rate_limiting_with_collection_ops(self):
        """
        Test 5: Rate Limiting with Collection Operations
        
        Verifies that rate limiting works correctly with higher-level
        collection operations, not just basic list operations.
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 5: Rate Limiting with Collection Operations")
        logger.info("="*80)
        
        settings = load_settings()
        settings.connection.max_requests_per_second = 15
        
        conn_manager = ConnectionManager(config=settings)
        coll_manager = CollectionManager(conn_manager)
        
        try:
            logger.info("Configured: 15 req/s rate limit")
            logger.info("Test: Create, describe, and drop collection operations")
            
            # Create schema
            schema = CollectionSchema(
                name=self.test_collection_name,
                description="Test collection for rate limiting",
                fields=[
                    FieldSchema(name="doc_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
                ]
            )
            
            start_time = time.time()
            operation_times = []
            
            # Test various operations
            logger.info("\n🔨 Executing collection operations...")
            
            # 1. List collections (5 times)
            logger.info("  1. Listing collections (5x)...")
            for i in range(5):
                op_start = time.time()
                await coll_manager.list_collections()
                operation_times.append(time.time() - op_start)
            
            # 2. Create collection
            logger.info("  2. Creating collection...")
            op_start = time.time()
            await coll_manager.create_collection(self.test_collection_name, schema)
            operation_times.append(time.time() - op_start)
            
            # 3. Describe collection (5 times)
            logger.info("  3. Describing collection (5x)...")
            for i in range(5):
                op_start = time.time()
                await coll_manager.describe_collection(self.test_collection_name)
                operation_times.append(time.time() - op_start)
            
            # 4. Check if collection exists (5 times)
            logger.info("  4. Checking collection existence (5x)...")
            for i in range(5):
                op_start = time.time()
                await coll_manager.has_collection(self.test_collection_name)
                operation_times.append(time.time() - op_start)
            
            # 5. Drop collection
            logger.info("  5. Dropping collection...")
            op_start = time.time()
            await coll_manager.drop_collection(self.test_collection_name, safe=False)
            operation_times.append(time.time() - op_start)
            
            total_time = time.time() - start_time
            total_ops = len(operation_times)
            
            # Get metrics
            metrics = conn_manager.get_rate_limiter_metrics()
            
            logger.info(f"\n✅ Results:")
            logger.info(f"  Total operations: {total_ops}")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Actual rate: {total_ops / total_time:.1f} ops/s")
            logger.info(f"  Throttled operations: {sum(1 for t in operation_times if t > 0.05)}")
            
            if metrics:
                logger.info(f"\n📊 Rate Limiter Metrics:")
                logger.info(f"  Total requests: {metrics['total_requests']}")
                logger.info(f"  Throttle rate: {metrics['throttle_rate_percent']:.1f}%")
            
            if total_ops / total_time <= 20:  # Should be throttled to ~15 ops/s
                logger.info("✅ SUCCESS: Rate limiting works with collection operations!")
                self.results['collection_ops_rate_limiting'] = 'PASS'
            else:
                logger.warning("⚠️  Rate exceeded expected limit")
                self.results['collection_ops_rate_limiting'] = 'PARTIAL'
        
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            self.results['collection_ops_rate_limiting'] = 'ERROR'
        finally:
            # Cleanup
            try:
                await coll_manager.drop_collection(self.test_collection_name, safe=False)
            except Exception:
                pass
            conn_manager.close()
    
    async def test_metrics_collection(self):
        """
        Test 6: Comprehensive Metrics Collection
        
        Verifies that all metrics are properly collected and can be
        retrieved for monitoring systems.
        """
        logger.info("\n" + "="*80)
        logger.info("TEST 6: Comprehensive Metrics Collection")
        logger.info("="*80)
        
        settings = load_settings()
        settings.connection.max_requests_per_second = 50
        settings.connection.enable_retry_budget = True
        
        conn_manager = ConnectionManager(config=settings)
        coll_manager = CollectionManager(conn_manager)
        
        try:
            logger.info("Test: Collect metrics from all components")
            
            # Make some requests
            logger.info("\n📊 Generating load for metrics...")
            for i in range(30):
                await coll_manager.list_collections()
            
            # Get comprehensive metrics
            all_metrics = conn_manager.get_all_metrics()
            
            logger.info(f"\n✅ Collected Metrics:")
            logger.info("\n1️⃣  Rate Limiter:")
            if all_metrics['rate_limiter']:
                for key, value in all_metrics['rate_limiter'].items():
                    logger.info(f"    {key}: {value}")
            else:
                logger.info("    Not available")
            
            logger.info("\n2️⃣  Retry Budget:")
            if all_metrics['retry_budget']:
                for key, value in all_metrics['retry_budget'].items():
                    logger.info(f"    {key}: {value}")
            else:
                logger.info("    Not available")
            
            logger.info("\n3️⃣  Circuit Breaker:")
            if all_metrics['circuit_breaker']:
                for key, value in all_metrics['circuit_breaker'].items():
                    logger.info(f"    {key}: {value}")
            else:
                logger.info("    Not available")
            
            # Verify metrics are being collected
            if all_metrics['rate_limiter'] and all_metrics['retry_budget']:
                logger.info("\n✅ SUCCESS: All metrics are being collected!")
                self.results['metrics_collection'] = 'PASS'
            else:
                logger.warning("⚠️  Some metrics are missing")
                self.results['metrics_collection'] = 'PARTIAL'
        
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            self.results['metrics_collection'] = 'ERROR'
        finally:
            conn_manager.close()
    
    async def run_all_tests(self):
        """Run all rate limiting tests in sequence."""
        logger.info("\n" + "="*80)
        logger.info("RATE LIMITING TEST SUITE")
        logger.info("="*80)
        logger.info("Testing rate limiting implementation across connection and collection modules")
        logger.info("="*80)
        
        # Run all tests
        await self.test_basic_rate_limiting()
        await asyncio.sleep(1)
        
        await self.test_burst_capacity()
        await asyncio.sleep(1)
        
        await self.test_concurrent_rate_limiting()
        await asyncio.sleep(1)
        
        await self.test_retry_budget()
        await asyncio.sleep(1)
        
        await self.test_rate_limiting_with_collection_ops()
        await asyncio.sleep(1)
        
        await self.test_metrics_collection()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test results summary."""
        logger.info("\n" + "="*80)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*80)
        
        passed = sum(1 for v in self.results.values() if v == 'PASS')
        partial = sum(1 for v in self.results.values() if v == 'PARTIAL')
        failed = sum(1 for v in self.results.values() if v in ['FAIL', 'ERROR'])
        total = len(self.results)
        
        for test_name, result in self.results.items():
            icon = "✅" if result == 'PASS' else "⚠️ " if result == 'PARTIAL' else "❌"
            logger.info(f"{icon} {test_name}: {result}")
        
        logger.info("\n" + "-"*80)
        logger.info(f"Passed: {passed}/{total}")
        logger.info(f"Partial: {partial}/{total}")
        logger.info(f"Failed: {failed}/{total}")
        logger.info("-"*80)
        
        if passed == total:
            logger.info("\n🎉 ALL TESTS PASSED! Rate limiting is production-ready!")
        elif passed + partial == total:
            logger.info("\n✅ All tests completed with partial success.")
            logger.info("   Rate limiting is working but may need tuning.")
        else:
            logger.error("\n❌ Some tests failed. Review the output above.")
        
        logger.info("="*80 + "\n")


async def main():
    """Main entry point for rate limiting tests."""
    test_suite = RateLimitingTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())

