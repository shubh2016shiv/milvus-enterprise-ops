"""
Basic Connection Example

Demonstrates how to establish a basic connection to Milvus server,
check server status, and properly close the connection.
"""

import sys
import os
import asyncio

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


async def main():
    """Main function to demonstrate basic connection."""
    print_section("Milvus Basic Connection Example")
    
    # Step 1: Initialize Connection Manager
    print_step(1, "Initialize Connection Manager")
    try:
        # Load configuration
        config = load_settings()
        print_info("Host", config.connection.host)
        print_info("Port", config.connection.port)
        print_info("Timeout", f"{config.connection.timeout}s")
        
        # Create connection manager
        conn_manager = ConnectionManager(config=config)
        print_success("Connection Manager initialized")
    except Exception as e:
        print_error(f"Failed to initialize: {e}")
        return
    
    # Step 2: Check Server Status
    print_step(2, "Check Milvus Server Status")
    try:
        is_available = conn_manager.check_server_status()
        if is_available:
            print_success("Milvus server is available and responsive")
        else:
            print_error("Milvus server is not responding")
            return
    except Exception as e:
        print_error(f"Status check failed: {e}")
        return
    
    # Step 3: Get Connection Metrics
    print_step(3, "Get Connection Metrics")
    try:
        metrics = conn_manager.get_all_metrics()
        
        if metrics.get('circuit_breaker'):
            cb_metrics = metrics['circuit_breaker']
            print_info("Circuit Breaker State", cb_metrics.get('state', 'N/A'))
            print_info("Circuit Breaker Failures", cb_metrics.get('failure_count', 0))
        
        if metrics.get('rate_limiter'):
            rl_metrics = metrics['rate_limiter']
            print_info("Rate Limiter Tokens", f"{rl_metrics.get('available_tokens', 0):.2f}")
        
        print_success("Retrieved connection metrics")
    except Exception as e:
        print_error(f"Failed to get metrics: {e}")
    
    # Step 4: Test Simple Operation
    print_step(4, "Execute Test Operation")
    try:
        def list_collections(conn_alias):
            from pymilvus import utility
            return utility.list_collections(using=conn_alias)
        
        collections = await conn_manager.execute_operation(list_collections)
        print_info("Collections found", len(collections))
        if collections:
            print_info("Collection names", ", ".join(collections[:5]))
        print_success("Test operation completed successfully")
    except Exception as e:
        print_error(f"Operation failed: {e}")
    
    # Step 5: Close Connection
    print_step(5, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed successfully")
    except Exception as e:
        print_error(f"Failed to close connection: {e}")
    
    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • ConnectionManager handles connection lifecycle")
    print("  • Server status check before operations")
    print("  • Metrics provide visibility into connection health")
    print("  • Always close connections properly")


if __name__ == "__main__":
    asyncio.run(main())
