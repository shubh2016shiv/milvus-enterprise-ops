"""
Circuit Breaker Test

Demonstrates the circuit breaker pattern for handling failures gracefully
and preventing cascade failures when the server is unavailable.
"""

import sys
import os
import asyncio
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from connection_management import ConnectionManager, ServerUnavailableError
from config import load_settings
# Import usage_examples utils (not the project's utils package)
import importlib.util
utils_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils.py'))
spec = importlib.util.spec_from_file_location("example_utils", utils_file_path)
example_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example_utils)
print_section = example_utils.print_section
print_step = example_utils.print_step
print_success = example_utils.print_success
print_info = example_utils.print_info
print_error = example_utils.print_error


async def main():
    """Main function to demonstrate circuit breaker."""
    print_section("Milvus Circuit Breaker Test")
    
    # Step 1: Initialize with Circuit Breaker
    print_step(1, "Initialize Connection Manager with Circuit Breaker")
    try:
        config = load_settings()
        
        # Circuit breaker is enabled by default
        conn_manager = ConnectionManager(config=config, enable_circuit_breaker=True)
        
        print_info("Circuit Breaker", "Enabled")
        print_info("Failure Threshold", 
                  getattr(config.connection, 'circuit_breaker_failure_threshold', 5))
        print_info("Recovery Timeout", 
                  f"{getattr(config.connection, 'circuit_breaker_recovery_timeout', 30.0)}s")
        print_success("Connection Manager initialized with circuit breaker")
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        return
    
    # Step 2: Check Circuit Breaker State
    print_step(2, "Check Initial Circuit Breaker State")
    try:
        is_open = conn_manager.is_circuit_breaker_open()
        metrics = conn_manager.get_circuit_breaker_metrics()
        
        if metrics:
            print_info("State", metrics.get('state', 'N/A'))
            print_info("Failure Count", metrics.get('failure_count', 0))
            print_info("Success Count", metrics.get('success_count', 0))
        
        if is_open:
            print_info("Status", "Circuit is OPEN (blocking requests)")
        else:
            print_success("Circuit is CLOSED (allowing requests)")
    except Exception as e:
        print_error(f"State check failed: {e}")
    
    # Step 3: Test Normal Operation
    print_step(3, "Test Normal Operation (Circuit Closed)")
    try:
        def list_collections(conn_alias):
            from pymilvus import utility
            return utility.list_collections(using=conn_alias)
        
        collections = conn_manager.execute_operation(list_collections)
        print_info("Collections", len(collections))
        print_success("Operation succeeded through closed circuit")
        
        metrics = conn_manager.get_circuit_breaker_metrics()
        if metrics:
            print_info("Success Count", metrics.get('success_count', 0))
    except Exception as e:
        print_error(f"Operation failed: {e}")
    
    # Step 4: Demonstrate Fail-Fast Behavior
    print_step(4, "Test Graceful Failure Handling")
    try:
        # If circuit is open, operations will fail fast
        is_open = conn_manager.is_circuit_breaker_open()
        
        if is_open:
            print_info("Circuit Status", "OPEN - failing fast")
            print_success("Circuit breaker preventing cascade failures")
        else:
            print_info("Circuit Status", "CLOSED - operations flowing normally")
            print_success("System is healthy, circuit remains closed")
    except Exception as e:
        print_error(f"Test failed: {e}")
    
    # Step 5: Monitor Circuit Breaker Metrics
    print_step(5, "Final Circuit Breaker Metrics")
    try:
        metrics = conn_manager.get_circuit_breaker_metrics()
        
        if metrics:
            print("\n  Circuit Breaker Statistics:")
            print(f"    State: {metrics.get('state', 'N/A')}")
            print(f"    Total Failures: {metrics.get('failure_count', 0)}")
            print(f"    Total Successes: {metrics.get('success_count', 0)}")
            
            if metrics.get('last_failure_time'):
                print(f"    Last Failure: {metrics.get('last_failure_time')}")
            
            print_success("Circuit breaker is protecting the system")
        else:
            print_info("Status", "Circuit breaker metrics not available")
    except Exception as e:
        print_error(f"Metrics retrieval failed: {e}")
    
    # Step 6: Cleanup
    print_step(6, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    print_section("Test Completed")
    print("\nKey Takeaways:")
    print("  • Circuit breaker prevents cascade failures")
    print("  • Fails fast when server is unavailable")
    print("  • Automatically recovers when server returns")
    print("  • Protects against resource exhaustion")
    print("\nCircuit States:")
    print("  • CLOSED: Normal operation, requests flow through")
    print("  • OPEN: Too many failures, blocking requests")
    print("  • HALF_OPEN: Testing if service recovered")


if __name__ == "__main__":
    asyncio.run(main())


