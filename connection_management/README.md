# Milvus Connection Management

This module provides robust, scalable connection management for Milvus, designed to handle high-volume concurrent access with built-in resilience.

## Key Features

- **Thread-safe connection pooling**: Efficiently manages connections for high-concurrency environments
- **Singleton pattern**: Ensures only one connection pool exists per application
- **Circuit breaker protection**: Prevents cascading failures and provides fast recovery from outages
- **Exponential backoff with jitter**: Handles transient connection issues more gracefully with adaptive retry delays
- **Connection health checks**: Validates connections before use to prevent stale or broken connections
- **Automatic retry**: Handles transient connection issues with configurable retry policies
- **Async support**: Provides both synchronous and asynchronous interfaces
- **Specialized error handling**: Detailed exception hierarchy for precise error handling
- **Resource management**: Proper cleanup of connections to prevent resource leaks
- **Production monitoring**: Comprehensive metrics for circuit breaker and connection pool health

## Usage Examples

### Basic Usage (using ConnectionManager)

```python
from Milvus_Ops.connection_management import ConnectionManager
from Milvus_Ops.config import load_settings

# Load configuration (or use default)
config = load_settings()

# Create connection manager
manager = ConnectionManager(config)

# Define an operation to perform (it will receive a connection alias)
def my_milvus_operation(conn_alias: str, collection_name: str):
    """Example operation using a Milvus connection."""
    # Use the connection alias with pymilvus operations
    # Example: connections.get_connection(conn_alias).list_collections()
    print(f"Performing operation with connection: {conn_alias} on collection: {collection_name}")
    # Replace with actual Milvus SDK calls
    return f"Operation on {collection_name} successful with {conn_alias}"

# Execute an operation with automatic retry and connection pooling
result = manager.execute_operation(my_milvus_operation, "my_collection")
print(f"Result: {result}")

# Don't forget to close when done to clean up the connection pool
manager.close()
```

### Direct Context Manager Usage (from ConnectionPool)

For more fine-grained control over connection acquisition and release, you can directly use the `MilvusConnectionPool` as a context manager:

```python
from Milvus_Ops.connection_management import MilvusConnectionPool
from Milvus_Ops.config import load_settings

config = load_settings()
pool = MilvusConnectionPool(config)

try:
    with pool.get_connection() as conn_alias:
        print(f"Directly using connection: {conn_alias}")
        # Perform Milvus operations using conn_alias
        # Example: connections.get_connection(conn_alias).load_collection("my_collection")
        pass
finally:
    # Ensure the pool is closed when done with all operations
    pool.close()
```

### Circuit Breaker Usage

The connection manager includes built-in circuit breaker protection to handle Milvus server outages gracefully:

```python
from Milvus_Ops.connection_management import ConnectionManager, CircuitBreakerConfig

# Create custom circuit breaker configuration
circuit_config = CircuitBreakerConfig(
    failure_threshold=3,        # Open circuit after 3 failures
    recovery_timeout=60.0,      # Wait 60 seconds before testing recovery
    half_open_success_threshold=2,  # Need 2 successes to close circuit
    max_half_open_requests=1    # Allow 1 concurrent test request
)

# Initialize with circuit breaker (enabled by default)
manager = ConnectionManager(enable_circuit_breaker=True)

def search_operation(conn_alias, query):
    # This operation will be protected by the circuit breaker
    return connections.get_connection(conn_alias).search(query)

try:
    result = manager.execute_operation(search_operation, "my_query")
except ServerUnavailableError as e:
    print(f"Milvus service temporarily unavailable: {e}")
    # Implement fallback logic here
except ConnectionError as e:
    print(f"Connection error: {e}")

# Check circuit breaker status
if manager.is_circuit_breaker_open():
    print("Circuit breaker is open - Milvus service is unavailable")

# Get circuit breaker metrics for monitoring
metrics = manager.get_circuit_breaker_metrics()
if metrics:
    print(f"Circuit state: {metrics['state']}")
    print(f"Success rate: {metrics['current_state']['success_rate_percent']}%")
```

### Monitoring and Metrics

```python
# Get comprehensive metrics for monitoring
metrics = manager.get_circuit_breaker_metrics()
if metrics:
    print("Circuit Breaker Metrics:")
    print(f"  State: {metrics['state']}")
    print(f"  Total Requests: {metrics['counters']['total_requests']}")
    print(f"  Success Rate: {metrics['current_state']['success_rate_percent']}%")
    print(f"  Fast Failures: {metrics['counters']['total_fast_failures']}")
    
    # Check if circuit breaker needs attention
    if metrics['current_state']['success_rate_percent'] < 95:
        print("WARNING: Low success rate detected")
    
    if metrics['state'] == 'open':
        print("ALERT: Circuit breaker is open - service unavailable")
```

```python
import asyncio
from Milvus_Ops.connection_management import ConnectionManager

async def main():
    manager = ConnectionManager()
    
    # Define an async operation to perform
    async def my_async_milvus_operation(conn_alias: str, collection_name: str):
        print(f"Performing async operation with connection: {conn_alias} on collection: {collection_name}")
        # Replace with actual async Milvus SDK calls if available or use sync calls in executor
        await asyncio.sleep(0.1) # Simulate async work
        return f"Async operation on {collection_name} successful with {conn_alias}"

    # Execute operation asynchronously
    result = await manager.execute_operation_async(my_async_milvus_operation, "my_async_collection")
    print(f"Async Result: {result}")
    
    # Check server status
    is_available = manager.check_server_status()
    print(f"Milvus server available: {is_available}")
    
    # Clean up
    manager.close()

# Run the async function
asyncio.run(main())
```

## Error Handling

```python
from Milvus_Ops.connection_management import (
    ConnectionError,
    MaxRetriesExceededError,
    ConnectionPoolExhaustedError
)

manager = ConnectionManager()

try:
    # Example operation that might fail
    def failing_operation(conn_alias):
        raise ConnectionError("Simulated connection failure")

    result = manager.execute_operation(failing_operation)
except ConnectionPoolExhaustedError:
    print("Connection pool exhausted - no connections available.")
except MaxRetriesExceededError:
    print("Operation failed after multiple retries due to persistent connection issues.")
except ConnectionError as e:
    print(f"General connection error: {e}")
finally:
    manager.close()
```

## Scaling Considerations

This module is designed to handle high-volume scenarios with:

- Connection pooling to reuse connections efficiently
- Thread safety for concurrent access
- Configurable pool size to match your workload
- **Exponential backoff with jitter** for resilience against transient failures and to prevent overwhelming the server.
- **Connection health checks** to ensure only active and valid connections are used.
- Proper resource cleanup to prevent leaks

For production environments with millions of users, consider:

1.  Adjusting `connection_pool_size` based on expected concurrent connections and monitoring actual usage.
2.  Tuning `timeout` settings for connection acquisition and Milvus operations to match your workload characteristics.
3.  Setting appropriate `retry_count`, `retry_interval` and understanding the impact of exponential backoff for your network conditions.
4.  Monitoring connection pool usage and Milvus server health to detect potential bottlenecks and ensure optimal performance.
5.  Implementing robust logging and alerting for connection-related errors and pool exhaustion events.
