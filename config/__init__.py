"""
Configuration Module

This module provides centralized configuration management for Milvus operations:
- Connection configuration
- Collection and schema defaults
- Index parameter presets
- Search parameter configurations
- Performance tuning settings
- Environment-specific configurations (dev, test, prod)
- Configuration validation and loading

Implements a flexible, environment-aware configuration system
with sensible defaults and comprehensive validation using Pydantic.
"""

from .settings import (
    MilvusSettings,
    load_settings,
    ConsistencyLevel,
    MetricType,
    IndexType
)

__all__ = [
    'MilvusSettings',
    'load_settings',
    'ConsistencyLevel',
    'MetricType',
    'IndexType'
]