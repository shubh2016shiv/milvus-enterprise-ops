"""
Comprehensive unit tests for index build progress tracking.

This module provides systematic testing of IndexBuildTracker and
IndexBuildTrackerRegistry classes in the index_operations.utils.progress module.
"""

import pytest

from milvus_ops.index_operations.models.entities import IndexBuildProgress, IndexState
from milvus_ops.index_operations.utils.progress import (
    IndexBuildTracker,
    IndexBuildTrackerRegistry,
    get_registry,
)

# ============================================================================
# Test IndexBuildTracker
# ============================================================================


@pytest.mark.unit
class TestIndexBuildTracker:
    """
    Test IndexBuildTracker class.

    Coverage: IndexBuildTracker initialization and methods.
    """

    def test_tracker_initialization(self):
        """
        Test IndexBuildTracker initialization.

        Coverage: IndexBuildTracker can be created with collection and field names.
        """
        tracker = IndexBuildTracker("test_collection", "embedding")
        assert tracker.collection_name == "test_collection"
        assert tracker.field_name == "embedding"
        assert tracker.total_rows is None
        assert tracker.state == IndexState.NONE

    def test_tracker_initialization_with_total_rows(self):
        """
        Test IndexBuildTracker initialization with total_rows.

        Coverage: IndexBuildTracker can be created with total_rows.
        """
        tracker = IndexBuildTracker("test_collection", "embedding", total_rows=10000)
        assert tracker.total_rows == 10000

    def test_tracker_start_tracking(self):
        """
        Test IndexBuildTracker start_tracking() method.

        Coverage: start_tracking() initializes tracking state.
        """
        tracker = IndexBuildTracker("test_collection", "embedding")
        tracker.start_tracking()
        assert tracker.state == IndexState.CREATING
        assert tracker.start_time is not None
        assert tracker.last_update_time is not None
        assert tracker.percentage == 0.0
        assert tracker.processed_rows == 0
        assert len(tracker.history) == 1

    def test_tracker_update_progress_with_processed_rows(self):
        """
        Test IndexBuildTracker update_progress() with processed_rows.

        Coverage: update_progress() updates progress using processed_rows.
        """
        tracker = IndexBuildTracker("test_collection", "embedding", total_rows=10000)
        tracker.start_tracking()
        tracker.update_progress(processed_rows=5000)
        assert tracker.processed_rows == 5000
        assert tracker.percentage == 50.0

    def test_tracker_update_progress_with_percentage(self):
        """
        Test IndexBuildTracker update_progress() with percentage.

        Coverage: update_progress() updates progress using percentage.
        """
        tracker = IndexBuildTracker("test_collection", "embedding", total_rows=10000)
        tracker.start_tracking()
        tracker.update_progress(percentage=75.0)
        assert tracker.percentage == 75.0
        assert tracker.processed_rows == 7500

    def test_tracker_update_progress_state_transition(self):
        """
        Test IndexBuildTracker update_progress() state transitions.

        Coverage: update_progress() can change state to CREATED.
        """
        tracker = IndexBuildTracker("test_collection", "embedding")
        tracker.start_tracking()
        tracker.update_progress(state=IndexState.CREATED, percentage=100.0)
        assert tracker.state == IndexState.CREATED
        assert tracker.percentage == 100.0

    def test_tracker_update_progress_failed_state(self):
        """
        Test IndexBuildTracker update_progress() with FAILED state.

        Coverage: update_progress() handles FAILED state with reason.
        """
        tracker = IndexBuildTracker("test_collection", "embedding")
        tracker.start_tracking()
        tracker.update_progress(
            state=IndexState.FAILED,
            failed_reason="Memory error",
            percentage=30.0,
        )
        assert tracker.state == IndexState.FAILED
        assert tracker.failed_reason == "Memory error"
        assert tracker.percentage == 30.0

    def test_tracker_update_progress_without_total_rows(self):
        """
        Test IndexBuildTracker update_progress() without total_rows.

        Coverage: update_progress() works without total_rows.
        """
        tracker = IndexBuildTracker("test_collection", "embedding")
        tracker.start_tracking()
        tracker.update_progress(percentage=50.0)
        assert tracker.percentage == 50.0
        # processed_rows should be None or 0 when total_rows is None
        assert tracker.processed_rows == 0 or tracker.processed_rows is None

    def test_tracker_estimate_completion_time_complete(self):
        """
        Test IndexBuildTracker estimate_completion_time() for completed build.

        Coverage: estimate_completion_time() returns None, 0.0 for CREATED state.
        """
        tracker = IndexBuildTracker("test_collection", "embedding")
        tracker.start_tracking()
        tracker.update_progress(state=IndexState.CREATED, percentage=100.0)
        completion_time, remaining = tracker.estimate_completion_time()
        assert completion_time is None
        assert remaining == 0.0

    def test_tracker_estimate_completion_time_failed(self):
        """
        Test IndexBuildTracker estimate_completion_time() for failed build.

        Coverage: estimate_completion_time() returns None, None for FAILED state.
        """
        tracker = IndexBuildTracker("test_collection", "embedding")
        tracker.start_tracking()
        tracker.update_progress(state=IndexState.FAILED, failed_reason="Error")
        completion_time, remaining = tracker.estimate_completion_time()
        assert completion_time is None
        assert remaining is None

    def test_tracker_estimate_completion_time_with_rows(self):
        """
        Test IndexBuildTracker estimate_completion_time() with total_rows.

        Coverage: estimate_completion_time() estimates using row progress.
        """
        tracker = IndexBuildTracker("test_collection", "embedding", total_rows=10000)
        tracker.start_tracking()
        # Simulate some progress
        import time

        time.sleep(0.01)  # Small delay to ensure time difference
        tracker.update_progress(processed_rows=5000)
        time.sleep(0.01)
        completion_time, remaining = tracker.estimate_completion_time()
        # Should have an estimate based on progress rate
        assert remaining is not None or completion_time is not None

    def test_tracker_estimate_completion_time_without_history(self):
        """
        Test IndexBuildTracker estimate_completion_time() without sufficient history.

        Coverage: estimate_completion_time() returns None with insufficient history.
        """
        tracker = IndexBuildTracker("test_collection", "embedding")
        tracker.start_tracking()
        completion_time, remaining = tracker.estimate_completion_time()
        # With only one history entry, should return None
        assert completion_time is None or remaining is None

    def test_tracker_get_current_progress(self):
        """
        Test IndexBuildTracker get_current_progress() method.

        Coverage: get_current_progress() returns IndexBuildProgress instance.
        """
        tracker = IndexBuildTracker("test_collection", "embedding", total_rows=10000)
        tracker.start_tracking()
        tracker.update_progress(processed_rows=5000, percentage=50.0)
        progress = tracker.get_current_progress()
        assert isinstance(progress, IndexBuildProgress)
        assert progress.collection_name == "test_collection"
        assert progress.field_name == "embedding"
        assert progress.percentage == 50.0
        assert progress.processed_rows == 5000

    def test_tracker_is_complete_true(self):
        """
        Test IndexBuildTracker is_complete() for completed build.

        Coverage: is_complete() returns True for CREATED state.
        """
        tracker = IndexBuildTracker("test_collection", "embedding")
        tracker.start_tracking()
        tracker.update_progress(state=IndexState.CREATED)
        assert tracker.is_complete() is True

    def test_tracker_is_complete_failed(self):
        """
        Test IndexBuildTracker is_complete() for failed build.

        Coverage: is_complete() returns True for FAILED state.
        """
        tracker = IndexBuildTracker("test_collection", "embedding")
        tracker.start_tracking()
        tracker.update_progress(state=IndexState.FAILED)
        assert tracker.is_complete() is True

    def test_tracker_is_complete_false(self):
        """
        Test IndexBuildTracker is_complete() for in-progress build.

        Coverage: is_complete() returns False for CREATING state.
        """
        tracker = IndexBuildTracker("test_collection", "embedding")
        tracker.start_tracking()
        assert tracker.is_complete() is False

    def test_tracker_history_trimming(self):
        """
        Test IndexBuildTracker history trimming.

        Coverage: History is trimmed when it exceeds 100 entries.
        """
        tracker = IndexBuildTracker("test_collection", "embedding")
        tracker.start_tracking()
        # Add many history entries
        for i in range(150):
            tracker.update_progress(percentage=i / 150.0 * 100)
        # History should be trimmed to first entry + last 50
        assert len(tracker.history) <= 51  # First entry + last 50


# ============================================================================
# Test IndexBuildTrackerRegistry
# ============================================================================


@pytest.mark.unit
class TestIndexBuildTrackerRegistry:
    """
    Test IndexBuildTrackerRegistry class.

    Coverage: IndexBuildTrackerRegistry initialization and methods.
    """

    def test_registry_initialization(self):
        """
        Test IndexBuildTrackerRegistry initialization.

        Coverage: IndexBuildTrackerRegistry initializes with empty trackers.
        """
        registry = IndexBuildTrackerRegistry()
        assert registry._trackers == {}

    def test_registry_register_build(self):
        """
        Test IndexBuildTrackerRegistry register_build() method.

        Coverage: register_build() creates and registers a tracker.
        """
        registry = IndexBuildTrackerRegistry()
        tracker = registry.register_build("test_collection", "embedding")
        assert isinstance(tracker, IndexBuildTracker)
        assert tracker.collection_name == "test_collection"
        assert tracker.field_name == "embedding"
        assert tracker.state == IndexState.CREATING

    def test_registry_register_build_with_total_rows(self):
        """
        Test IndexBuildTrackerRegistry register_build() with total_rows.

        Coverage: register_build() can specify total_rows.
        """
        registry = IndexBuildTrackerRegistry()
        tracker = registry.register_build("test_collection", "embedding", total_rows=10000)
        assert tracker.total_rows == 10000

    def test_registry_get_tracker(self):
        """
        Test IndexBuildTrackerRegistry get_tracker() method.

        Coverage: get_tracker() returns registered tracker.
        """
        registry = IndexBuildTrackerRegistry()
        tracker1 = registry.register_build("test_collection", "embedding")
        tracker2 = registry.get_tracker("test_collection", "embedding")
        assert tracker1 is tracker2

    def test_registry_get_tracker_none(self):
        """
        Test IndexBuildTrackerRegistry get_tracker() for non-existent tracker.

        Coverage: get_tracker() returns None for non-existent tracker.
        """
        registry = IndexBuildTrackerRegistry()
        tracker = registry.get_tracker("test_collection", "embedding")
        assert tracker is None

    def test_registry_update_progress(self):
        """
        Test IndexBuildTrackerRegistry update_progress() method.

        Coverage: update_progress() updates registered tracker.
        """
        registry = IndexBuildTrackerRegistry()
        tracker = registry.register_build("test_collection", "embedding")
        registry.update_progress("test_collection", "embedding", processed_rows=5000)
        assert tracker.processed_rows == 5000

    def test_registry_update_progress_no_tracker(self):
        """
        Test IndexBuildTrackerRegistry update_progress() without tracker.

        Coverage: update_progress() handles missing tracker gracefully.
        """
        registry = IndexBuildTrackerRegistry()
        # Should not raise error
        registry.update_progress("test_collection", "embedding", processed_rows=5000)

    def test_registry_get_progress(self):
        """
        Test IndexBuildTrackerRegistry get_progress() method.

        Coverage: get_progress() returns progress from registered tracker.
        """
        registry = IndexBuildTrackerRegistry()
        registry.register_build("test_collection", "embedding", total_rows=10000)
        registry.update_progress("test_collection", "embedding", processed_rows=5000)
        progress = registry.get_progress("test_collection", "embedding")
        assert isinstance(progress, IndexBuildProgress)
        assert progress.processed_rows == 5000

    def test_registry_get_progress_none(self):
        """
        Test IndexBuildTrackerRegistry get_progress() for non-existent tracker.

        Coverage: get_progress() returns None for non-existent tracker.
        """
        registry = IndexBuildTrackerRegistry()
        progress = registry.get_progress("test_collection", "embedding")
        assert progress is None

    def test_registry_remove_tracker(self):
        """
        Test IndexBuildTrackerRegistry remove_tracker() method.

        Coverage: remove_tracker() removes tracker from registry.
        """
        registry = IndexBuildTrackerRegistry()
        registry.register_build("test_collection", "embedding")
        assert registry.get_tracker("test_collection", "embedding") is not None
        registry.remove_tracker("test_collection", "embedding")
        assert registry.get_tracker("test_collection", "embedding") is None

    def test_registry_remove_tracker_nonexistent(self):
        """
        Test IndexBuildTrackerRegistry remove_tracker() for non-existent tracker.

        Coverage: remove_tracker() handles missing tracker gracefully.
        """
        registry = IndexBuildTrackerRegistry()
        # Should not raise error
        registry.remove_tracker("test_collection", "embedding")

    def test_registry_get_active_builds(self):
        """
        Test IndexBuildTrackerRegistry get_active_builds() method.

        Coverage: get_active_builds() returns only active builds.
        """
        registry = IndexBuildTrackerRegistry()
        registry.register_build("test_collection", "embedding1")
        tracker2 = registry.register_build("test_collection", "embedding2")
        tracker2.update_progress(state=IndexState.CREATED)
        active_builds = registry.get_active_builds()
        assert len(active_builds) == 1
        assert active_builds[0].field_name == "embedding1"

    def test_registry_get_active_builds_empty(self):
        """
        Test IndexBuildTrackerRegistry get_active_builds() with no active builds.

        Coverage: get_active_builds() returns empty list when no active builds.
        """
        registry = IndexBuildTrackerRegistry()
        tracker = registry.register_build("test_collection", "embedding")
        tracker.update_progress(state=IndexState.CREATED)
        active_builds = registry.get_active_builds()
        assert len(active_builds) == 0

    def test_registry_get_all_builds(self):
        """
        Test IndexBuildTrackerRegistry get_all_builds() method.

        Coverage: get_all_builds() returns all builds regardless of state.
        """
        registry = IndexBuildTrackerRegistry()
        registry.register_build("test_collection", "embedding1")
        tracker2 = registry.register_build("test_collection", "embedding2")
        tracker2.update_progress(state=IndexState.CREATED)
        all_builds = registry.get_all_builds()
        assert len(all_builds) == 2

    def test_registry_get_all_builds_empty(self):
        """
        Test IndexBuildTrackerRegistry get_all_builds() with no builds.

        Coverage: get_all_builds() returns empty list when no builds.
        """
        registry = IndexBuildTrackerRegistry()
        all_builds = registry.get_all_builds()
        assert len(all_builds) == 0

    def test_registry_multiple_collections(self):
        """
        Test IndexBuildTrackerRegistry with multiple collections.

        Coverage: Registry handles multiple collections correctly.
        """
        registry = IndexBuildTrackerRegistry()
        tracker1 = registry.register_build("collection1", "embedding")
        tracker2 = registry.register_build("collection2", "embedding")
        assert tracker1 is not tracker2
        assert registry.get_tracker("collection1", "embedding") is tracker1
        assert registry.get_tracker("collection2", "embedding") is tracker2


# ============================================================================
# Test get_registry Function
# ============================================================================


@pytest.mark.unit
class TestGetRegistry:
    """
    Test get_registry() function.

    Coverage: get_registry() singleton pattern.
    """

    def test_get_registry_singleton(self):
        """
        Test get_registry() returns singleton instance.

        Coverage: get_registry() returns the same instance on multiple calls.
        """
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_get_registry_type(self):
        """
        Test get_registry() returns correct type.

        Coverage: get_registry() returns IndexBuildTrackerRegistry instance.
        """
        registry = get_registry()
        assert isinstance(registry, IndexBuildTrackerRegistry)
