"""
Partition Statistics Example

Demonstrates how to retrieve and analyze partition statistics using the
Partition Operations module. Shows how to monitor partition size, entity counts,
and resource usage for optimization and capacity planning.
"""

import sys
import os
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from connection_management import ConnectionManager
from partition_operations import PartitionManager
from collection_operations import CollectionManager
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
print_note = example_utils.print_note


COLLECTION_NAME = "test_partition_collection"
VECTOR_DIM = 128


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable format."""
    if bytes_value == 0:
        return "0 B"

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return ".1f"
        bytes_value /= 1024.0
    return ".1f"


def print_partition_stats(stats):
    """Print formatted partition statistics."""
    print(f"\n  Partition: {stats.name}")
    print(f"  Entity count: {stats.row_count:,}")
    print(f"  Memory size: {format_bytes(stats.memory_size)}")
    print(f"  Disk size: {format_bytes(stats.disk_size)}")
    print(f"  Index size: {format_bytes(stats.index_size)}")
    print(f"  Total size: {format_bytes(stats.total_size)}")
    print(f"  Segments: {stats.num_segments}")

    if stats.num_segments > 0:
        avg_size = stats.average_segment_size
        print(f"  Average segment size: {format_bytes(avg_size)}")

    if stats.is_empty:
        print("  Status: EMPTY (no data)")
    else:
        print("  Status: ACTIVE (contains data)")


async def get_existing_test_partitions(partition_manager: PartitionManager):
    """Get existing partitions for statistics demonstration."""
    try:
        all_partitions = await partition_manager.list_partitions(COLLECTION_NAME)

        # Filter out _default and get some test partitions
        test_partitions = [p for p in all_partitions if p != "_default"][:4]  # Take up to 4 partitions

        if len(test_partitions) < 2:
            print_info("Setup", "Need at least 2 partitions for comparison. Using available partitions.")
            test_partitions = all_partitions[:4]  # Include _default if needed

        print_info("Using partitions", f"{', '.join(test_partitions)}")
        return test_partitions

    except Exception as e:
        print_error(f"Failed to get existing partitions: {e}")
        return []


async def analyze_partition_sizes(partition_manager: PartitionManager, partition_names: list):
    """Analyze and compare partition sizes."""
    print("\n  === Partition Size Analysis ===")

    stats_list = []
    total_entities = 0
    total_size = 0

    for partition_name in partition_names:
        try:
            stats = await partition_manager.get_partition_stats(
                collection_name=COLLECTION_NAME,
                partition_name=partition_name,
                timeout=30.0
            )
            stats_list.append((partition_name, stats))
            total_entities += stats.row_count
            total_size += stats.total_size

        except Exception as e:
            print_error(f"Failed to get stats for '{partition_name}': {e}")

    # Sort by size for better display
    stats_list.sort(key=lambda x: x[1].total_size, reverse=True)

    print(f"\n  Summary across {len(stats_list)} partitions:")
    print(f"  Total entities: {total_entities:,}")
    print(f"  Total size: {format_bytes(total_size)}")
    print(".1f" if total_entities > 0 else "  Average entities per partition: 0")

    print("\n  Partition breakdown (sorted by size):")
    for partition_name, stats in stats_list:
        size_pct = (stats.total_size / total_size * 100) if total_size > 0 else 0
        entity_pct = (stats.row_count / total_entities * 100) if total_entities > 0 else 0
        print("20"
              "6,"
              "5.1f"
              "5.1f")


async def main():
    """Main function to demonstrate partition statistics."""
    print_section("Partition Statistics Example")

    # Step 1: Initialize Managers
    print_step(1, "Initialize Managers")
    try:
        config = load_settings()
        conn_manager = ConnectionManager(config=config)
        coll_manager = CollectionManager(conn_manager)
        partition_manager = PartitionManager(conn_manager)
        print_success("Managers initialized")
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        return

    # Step 2: Verify Collection Exists
    print_step(2, "Verify Collection Exists")
    try:
        exists = await coll_manager.has_collection(COLLECTION_NAME)
        if not exists:
            print_error(f"Collection '{COLLECTION_NAME}' does not exist")
            print_note("Run create_partition.py first to create the test collection")
            conn_manager.close()
            return
        print_success(f"Collection '{COLLECTION_NAME}' exists")
    except Exception as e:
        print_error(f"Failed to check collection: {e}")
        conn_manager.close()
        return

    # Step 3: Get Existing Partitions for Analysis
    print_step(3, "Get Existing Partitions for Analysis")
    test_partitions = await get_existing_test_partitions(partition_manager)

    # Step 4: Load Collection for Statistics
    print_step(4, "Load Collection for Statistics")
    try:
        await coll_manager.load_collection(COLLECTION_NAME)
        print_success("Collection loaded")
    except Exception as e:
        print_error(f"Failed to load collection: {e}")
        conn_manager.close()
        return

    # Step 5: Get Statistics for Individual Partitions
    print_step(5, "Get Statistics for Individual Partitions")
    for partition_name in test_partitions:
        try:
            print(f"\n  Getting statistics for '{partition_name}'...")
            stats = await partition_manager.get_partition_stats(
                collection_name=COLLECTION_NAME,
                partition_name=partition_name,
                timeout=30.0
            )
            print_partition_stats(stats)

        except Exception as e:
            print_error(f"Failed to get stats for '{partition_name}': {e}")

    # Step 6: Analyze Partition Sizes
    print_step(6, "Analyze Partition Sizes")
    await analyze_partition_sizes(partition_manager, test_partitions)

    # Step 7: Compare Partition Statistics
    print_step(7, "Compare Partition Statistics")
    try:
        if len(test_partitions) >= 2:
            # Compare two partitions
            partition1, partition2 = test_partitions[:2]

            stats1 = await partition_manager.get_partition_stats(COLLECTION_NAME, partition1)
            stats2 = await partition_manager.get_partition_stats(COLLECTION_NAME, partition2)

            print(f"\n  Comparing '{partition1}' vs '{partition2}':")
            print(f"  {partition1}: {stats1.row_count:,} entities, {format_bytes(stats1.total_size)}")
            print(f"  {partition2}: {stats2.row_count:,} entities, {format_bytes(stats2.total_size)}")

            size_diff = abs(stats1.total_size - stats2.total_size)
            entity_diff = abs(stats1.row_count - stats2.row_count)

            print(f"  Size difference: {format_bytes(size_diff)}")
            print(f"  Entity difference: {entity_diff:,}")
        else:
            print_info("Comparison", "Need at least 2 partitions for comparison")

    except Exception as e:
        print_error(f"Failed to compare partition statistics: {e}")

    # Step 8: Check Default Partition Statistics
    print_step(8, "Check Default Partition Statistics")
    try:
        print("\n  Default partition statistics:")
        default_stats = await partition_manager.get_partition_stats(
            collection_name=COLLECTION_NAME,
            partition_name="_default",
            timeout=30.0
        )
        print_partition_stats(default_stats)

        if default_stats.is_empty:
            print_note("The _default partition is empty (as expected)")
        else:
            print_note("The _default partition contains data")

    except Exception as e:
        print_error(f"Failed to get default partition stats: {e}")

    # Step 9: Performance Considerations
    print_step(9, "Performance Considerations")
    try:
        import time

        # Test statistics retrieval performance
        partitions_to_test = test_partitions[:3]  # Test first 3 partitions

        start_time = time.time()
        for _ in range(5):  # Get stats 5 times for each partition
            for partition_name in partitions_to_test:
                await partition_manager.get_partition_stats(COLLECTION_NAME, partition_name)

        elapsed = time.time() - start_time
        avg_time = elapsed / (5 * len(partitions_to_test))

        print(".3f")
        print(".3f")
        print_note("Statistics retrieval is typically fast")

    except Exception as e:
        print_error(f"Performance test failed: {e}")

    # Step 10: Demonstrate Error Handling
    print_step(10, "Demonstrate Error Handling")
    try:
        # Try to get stats for non-existent partition
        stats = await partition_manager.get_partition_stats(
            collection_name=COLLECTION_NAME,
            partition_name="non_existent_partition"
        )
        print_info("Non-existent partition", "Stats retrieval returned data")
    except Exception as e:
        print_info("Expected error", f"Getting stats for non-existent partition: {type(e).__name__}")

    try:
        # Try to get stats from non-existent collection
        stats = await partition_manager.get_partition_stats(
            collection_name="non_existent_collection",
            partition_name="test"
        )
        print_info("Non-existent collection", "Stats retrieval returned data")
    except Exception as e:
        print_info("Expected error", f"Getting stats from non-existent collection: {type(e).__name__}")

    # Step 11: Cleanup
    print_step(11, "Close Connection")
    try:
        conn_manager.close()
        print_success("Connection closed")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")

    print_section("Example Completed")
    print("\nKey Takeaways:")
    print("  • Use get_partition_stats() to monitor partition size and entity counts")
    print("  • Statistics help with capacity planning and optimization decisions")
    print("  • Empty partitions show is_empty=True")
    print("  • Track size changes after data operations")
    print("  • Default partition (_default) exists in every collection")
    print("  • Statistics retrieval is fast and suitable for monitoring")
    print("  • Use formatted byte display for human-readable sizes")


if __name__ == "__main__":
    asyncio.run(main())
