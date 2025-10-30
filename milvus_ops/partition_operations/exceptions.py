"""
User-friendly exceptions for partition operations.

Provides clear, actionable error messages to help users
understand and resolve issues quickly.
"""


class PartitionError(Exception):
    """
    Base exception for partition operations.

    Provides clear context about what went wrong and where.
    """

    def __init__(
        self,
        message: str,
        collection_name: str = None,
        partition_name: str = None
    ):
        self.collection_name = collection_name
        self.partition_name = partition_name

        # Build contextual message
        context_parts = []
        if collection_name:
            context_parts.append(f"collection '{collection_name}'")
        if partition_name:
            context_parts.append(f"partition '{partition_name}'")

        if context_parts:
            full_message = f"{' in '.join(context_parts)}: {message}"
        else:
            full_message = message

        super().__init__(full_message)


class PartitionNotFoundError(PartitionError):
    """
    Raised when a partition doesn't exist.

    The partition may have been deleted or never created.
    Check the partition name and create it if needed.
    """

    def __init__(self, partition_name: str, collection_name: str):
        message = (
            f"Partition not found. Check the name or create the partition first."
        )
        super().__init__(message, collection_name, partition_name)


class PartitionAlreadyExistsError(PartitionError):
    """
    Raised when trying to create an existing partition.

    The partition already exists - no action needed.
    Use create_partition_if_not_exists() for idempotent creation.
    """

    def __init__(self, partition_name: str, collection_name: str):
        message = "Partition already exists"
        super().__init__(message, collection_name, partition_name)


class InvalidPartitionNameError(PartitionError):
    """
    Raised when a partition name doesn't meet requirements.

    Partition names must:
    - Be 1-255 characters long
    - Contain only letters, numbers, underscores, and hyphens
    - Not be reserved names like '_default'
    - Not start/end with special characters
    """

    def __init__(self, partition_name: str, reason: str):
        message = f"Invalid partition name: {reason}"
        super().__init__(message, partition_name=partition_name)


class PartitionOperationError(PartitionError):
    """
    Raised when a partition operation fails.

    This could be due to network problems, server issues, or timeouts.
    Check your connection and try again.
    """

    def __init__(
        self,
        operation: str,
        collection_name: str,
        partition_name: str = None,
        reason: str = None
    ):
        message = f"Failed to {operation}"
        if reason:
            message += f": {reason}"
        super().__init__(message, collection_name, partition_name)