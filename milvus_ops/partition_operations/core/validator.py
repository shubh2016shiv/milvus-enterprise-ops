"""
Partition validator for Milvus operations.

Provides validation logic for partition names to ensure they meet
all requirements before attempting operations.
"""

import logging
import re

logger = logging.getLogger(__name__)


class PartitionValidator:
    """
    Validates partition names according to Milvus constraints.

    Performs pre-flight checks to ensure partition names are valid,
    providing clear error feedback without requiring server roundtrips.
    """

    # Maximum length for partition names
    MAX_NAME_LENGTH = 255

    # Reserved partition names
    RESERVED_NAMES = frozenset({"_default", "default"})

    # Pattern for valid partition names (alphanumeric, underscores, hyphens)
    VALID_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

    @classmethod
    async def validate_partition_name(cls, partition_name: str) -> tuple[bool, list[str]]:
        """
        Validates a partition name according to Milvus constraints.

        Args:
            partition_name: The partition name to validate

        Returns:
            Tuple of (is_valid, error_list)
            - is_valid: True if name is valid, False otherwise
            - error_list: List of validation errors (empty if valid)
        """
        errors = []

        # Check for empty or None
        if not partition_name or not partition_name.strip():
            errors.append("Partition name cannot be empty or whitespace")
            return False, errors

        # Normalize for validation
        name = partition_name.strip()

        # Check length
        if len(name) > cls.MAX_NAME_LENGTH:
            errors.append(
                f"Partition name exceeds maximum length of {cls.MAX_NAME_LENGTH} characters "
                f"(got {len(name)})"
            )

        # Check for reserved names
        if name.lower() in cls.RESERVED_NAMES:
            errors.append(f"'{name}' is a reserved name and cannot be used")

        # Check pattern (alphanumeric, underscores, hyphens only)
        if not cls.VALID_NAME_PATTERN.match(name):
            errors.append("Name can only contain letters, numbers, underscores, and hyphens")

        # Check for leading/trailing special characters (best practice)
        if name and name[0] in ("_", "-"):
            errors.append("Name should not start with underscore or hyphen")
        if name and name[-1] in ("_", "-"):
            errors.append("Name should not end with underscore or hyphen")

        # Check for consecutive underscores (can cause issues)
        if "__" in name:
            errors.append("Name cannot contain consecutive underscores")

        # Check if original had whitespace
        if partition_name != name:
            errors.append("Name cannot have leading or trailing whitespace")

        return len(errors) == 0, errors

    @classmethod
    def sanitize_partition_name(cls, name: str) -> str:
        """
        Attempts to sanitize a partition name to make it valid.

        This is a best-effort approach - validation should still be done
        after sanitization.

        Args:
            name: The name to sanitize

        Returns:
            Sanitized name
        """
        if not name:
            return ""

        # Strip whitespace
        name = name.strip()

        # Replace spaces with underscores
        name = name.replace(" ", "_")

        # Remove invalid characters
        name = "".join(c for c in name if c.isalnum() or c in ("_", "-"))

        # Remove leading/trailing special characters
        name = name.strip("_-")

        # Replace consecutive underscores
        while "__" in name:
            name = name.replace("__", "_")

        # Truncate to max length
        if len(name) > cls.MAX_NAME_LENGTH:
            name = name[: cls.MAX_NAME_LENGTH].rstrip("_-")

        return name
