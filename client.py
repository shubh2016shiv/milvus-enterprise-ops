"""
Milvus Client

This module provides the main client interface for Milvus operations,
integrating all the functionality from the submodules.
"""

from typing import Dict, Any, List, Optional, Union
import logging
from pathlib import Path

from .config import MilvusSettings, load_settings
from .exceptions import ConnectionError, ConfigurationError

# Logger setup
logger = logging.getLogger(__name__)


class MilvusClient:
    """
    Main client interface for Milvus operations.
    
    This class provides a unified interface to all Milvus operations,
    integrating functionality from the various submodules.
    """
    
    def __init__(self, config: Optional[Union[MilvusSettings, str, Path]] = None):
        """
        Initialize the Milvus client.
        
        Args:
            config: Either a MilvusSettings object or a path to a config YAML file.
                   If None, default configuration will be used.
        """
        # Load configuration
        if config is None:
            self.config = load_settings()
        elif isinstance(config, (str, Path)):
            self.config = load_settings(str(config))
        elif isinstance(config, MilvusSettings):
            self.config = config
        else:
            raise ConfigurationError("Invalid configuration type. Expected MilvusSettings, str, Path, or None.")
        
        # Initialize connection
        self._initialize_connection()
        
        # Initialize submodules
        self._initialize_submodules()
        
        logger.info("MilvusClient initialized successfully")
    
    def _initialize_connection(self):
        """Initialize connection to Milvus server"""
        try:
            # Connection initialization logic will be implemented here
            # This will use the connection_management module
            pass
        except Exception as e:
            logger.error(f"Failed to connect to Milvus server: {e}")
            raise ConnectionError(f"Failed to connect to Milvus server: {e}")
    
    def _initialize_submodules(self):
        """Initialize all submodules"""
        # These will be initialized with actual implementations
        # from their respective modules
        self.collection = None  # Will be collection_operations.CollectionManager
        self.index = None       # Will be index_operations.IndexManager
        self.partition = None   # Will be partition_operations.PartitionManager
        self.query = None       # Will be data_query_operations.QueryManager
        self.insert = None      # Will be data_insertion_operations.InsertionManager
        self.modify = None      # Will be data_modification_operations.ModificationManager
        self.backup = None      # Will be backup_recovery.BackupManager
        self.monitor = None     # Will be monitoring.MonitoringManager
    
    def close(self):
        """Close the client and release all resources"""
        # Connection cleanup logic will be implemented here
        logger.info("MilvusClient connection closed")