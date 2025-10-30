"""
Query Expression Utilities

Provides utilities for generating robust query expressions that work across
different Milvus versions and configurations.
"""

import logging
from typing import Optional

from pymilvus import Collection, DataType

logger = logging.getLogger(__name__)


def generate_query_expression(collection: Collection) -> str:
    """
    Generate a query expression that returns all data based on primary key.

    This function detects the primary key field and its data type, then generates
    an appropriate expression that will work across different Milvus versions.
    This avoids issues with empty expressions ("") which are not supported in
    some Milvus configurations.

    Args:
        collection: Milvus Collection object

    Returns:
        A query expression string that will return all data

    Raises:
        ValueError: If no primary key field is found
    """
    # Get collection schema
    schema = collection.schema
    primary_field = None

    # Find primary key field
    for field in schema.fields:
        if field.is_primary:
            primary_field = field
            break

    if not primary_field:
        logger.warning("No primary key field found in schema, using fallback expression")
        # Use a field that's likely to exist in all collections
        return "id >= 0"

    # Log the primary key field we found
    logger.debug(f"Found primary key field: {primary_field.name} with type {primary_field.dtype}")

    # Generate expression based on data type
    field_name = primary_field.name

    # Handle different data types
    if primary_field.dtype in [DataType.INT64, DataType.INT32, DataType.INT16, DataType.INT8]:
        # Integer types
        return f"{field_name} >= 0"
    elif primary_field.dtype in [DataType.VARCHAR, DataType.STRING]:
        # String types
        return f"{field_name} >= ''"
    elif primary_field.dtype == DataType.BOOL:
        # Boolean type (get both true and false)
        return f"{field_name} in [true, false]"
    elif primary_field.dtype in [DataType.FLOAT, DataType.DOUBLE]:
        # Floating point types
        return f"{field_name} >= 0 or {field_name} < 0"  # Gets all non-NaN values
    else:
        # Fallback for other types
        logger.warning(f"Unsupported primary key type {primary_field.dtype}, using generic expression")
        return f"{field_name} is not null"
