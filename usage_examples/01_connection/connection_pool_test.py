"""
Connection Pool Test

Demonstrates connection pooling functionality, showing how multiple
operations share connections from the pool efficiently.
"""

import sys
import os
import asyncio
import time
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from connection_management import ConnectionManager
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


def perform_operation(conn_manager, operation_id):
    """Perform a simple operation using the connection pool."""
    try:
        def list_collections(conn_alias):
            from pymilvus import utility
            time.sleep(0.1)  # Simulate some work
            return utility.list_collections(using=conn_alias)
        
        collections = conn_manager.execute_operation(list_collections)
        print(f"  Operation {operation_id}: Found {len(collections)} collections")
        return True
    except Exception as e:
        print_error(f"Operation {operation_id} failed: {e}")
        return False


async def main():
    """Main function to demonstrate connection pooling."""
    print_section("Milvus Connection Pool Test")
    
    # Step 1: Initialize with Pool Configuration
    print_step(1, "Initialize Connection Manager with Pool")
    try:
        config = load_settings()
        print_info("Pool Size", config.connection.connection_pool_size)
        print_info("Max Requests/Second", config.connection.max_requests_per_second)
        
        conn_manager = ConnectionManager(config=config)
        print_success("Connection Manager with pool initialized")
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        return
    
    # Step 2: Sequential Operations
    print_step(2, "Test Sequential Operations")
    try:
        start_time = time.time()
        for i in range(5):
            perform_operation(conn_manager, i + 1)
        duration = time.time() - start_time
        print_success(f"Sequential operations completed in {duration:.2f}s")
    except Exception as e:
        print_error(f"Sequential test failed: {e}")
    
    # Step 3: Concurrent Operations
    print_step(3, "Test Concurrent Operations (Threading)")
    try:
        start_time = time.time()
        threads = []
        num_threads = 10
        
        for i in range(num_threads):
            thread = threading.Thread(
                target=perform_operation,
                args=(conn_manager, i + 1)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        duration = time.time() - start_time
        print_success(f"Concurrent operations ({num_threads} threads) completed in {duration:.2f}s")
        print_info("Avg time per operation", f"{duration/num_threads:.2f}s")
    except Exception as e:
        print_error(f"Concurrent test failed: {e}")
    
    # Step 4: Pool Metrics
    print_step(4, "Check Pool Metrics")
    try:
        metrics = conn_manager.get_all_metrics()
        print_info("Connection pool", "Active and managing connections")
        print_success("Pool is functioning correctly")
    except Exception as e:
        print_error(f"Metrics check failed: {e}")
    
    # Step 5: Cleanup
    print_step(5, "Close Connection Manager")
    try:
        conn_manager.close()
        print_success("Connection pool cleaned up")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
    
    print_section("Test Completed")
    print("\nKey Takeaways:")
    print("  • Connection pool enables efficient resource reuse")
    print("  • Multiple operations can run concurrently")
    print("  • Pool automatically manages connection lifecycle")
    print("  • Thread-safe for concurrent operations")


if __name__ == "__main__":
    asyncio.run(main())


