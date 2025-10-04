"""
Pydantic Settings for Milvus Operations

This module provides strongly-typed configuration settings using Pydantic,
with support for environment variables and YAML configuration files.
"""

from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pathlib import Path
import os

from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_yaml import to_yaml_str, to_yaml_file


class ConsistencyLevel(str, Enum):
    """
    Milvus consistency level options that determine the trade-off between consistency and performance.
    
    The consistency level affects how reads and writes are synchronized across distributed nodes:
    - Higher consistency levels provide stronger guarantees but may impact performance
    - Lower consistency levels offer better performance but weaker guarantees
    """
    STRONG = "Strong"  # Strongest consistency guarantee; ensures all operations are fully synchronized across all nodes before returning
    BOUNDED = "Bounded"  # Provides a time-bounded staleness guarantee; reads may return slightly stale data within defined bounds
    EVENTUALLY = "Eventually"  # Weakest consistency but highest performance; eventual consistency across all nodes
    SESSION = "Session"  # Provides read-your-writes consistency within a single client session


class MetricType(str, Enum):
    """
    Milvus metric type options for measuring distance/similarity between vectors.
    
    The choice of metric type significantly impacts search results and should match your embedding model:
    - Different metrics are suitable for different types of vector data and use cases
    - The metric type must be compatible with your index type
    - Some metrics require normalized vectors (e.g., COSINE)
    """
    L2 = "L2"  # Euclidean distance: Measures straight-line distance between vectors; lower values = more similar
    IP = "IP"  # Inner Product: Dot product between vectors; higher values = more similar; vectors should be normalized
    COSINE = "COSINE"  # Cosine similarity: Measures angle between vectors; higher values = more similar; best for semantic similarity
    HAMMING = "HAMMING"  # Hamming distance: Counts differing bits between binary vectors; lower values = more similar
    JACCARD = "JACCARD"  # Jaccard similarity: Ratio of intersection to union; higher values = more similar; used for sets


class IndexType(str, Enum):
    """
    Milvus index type options that determine search algorithm, performance characteristics, and memory usage.
    
    The choice of index type involves trade-offs between:
    - Search speed: How quickly results can be returned
    - Search accuracy: How precise the results are compared to exact search
    - Build time: How long it takes to create the index
    - Memory usage: How much RAM the index requires
    """
    FLAT = "FLAT"  # Brute-force exact search; 100% accuracy but slowest; suitable for small datasets or when perfect recall is required
    IVF_FLAT = "IVF_FLAT"  # Inverted file with flat quantization; good balance of accuracy and speed; uses clustering to narrow search space
    IVF_SQ8 = "IVF_SQ8"  # IVF with scalar quantization (8-bit); reduces memory by quantizing vectors; slight accuracy loss but better memory efficiency
    IVF_PQ = "IVF_PQ"  # IVF with product quantization; significant memory reduction; suitable for very large datasets with acceptable accuracy loss
    HNSW = "HNSW"  # Hierarchical Navigable Small World; excellent performance-accuracy trade-off; fast index building; suitable for dynamic data
    ANNOY = "ANNOY"  # Approximate Nearest Neighbors Oh Yeah; tree-based approach; good for memory-constrained environments
    SPARSE_INVERTED_INDEX = "SPARSE_INVERTED_INDEX"  # Specialized for sparse vectors; efficient for keyword/token-based search; used in hybrid search


class ConnectionSettings(BaseSettings):
    """
    Connection settings for establishing and maintaining connections to Milvus server.
    
    These settings control how the client connects to the Milvus server, including:
    - Server location and authentication
    - Connection security and timeout behavior
    - Connection pooling for performance optimization
    - Retry behavior for resilience against transient failures
    """
    host: str = Field("localhost", env="MILVUS_HOST", 
                     description="Hostname or IP address of the Milvus server")
    port: str = Field("19530", env="MILVUS_PORT", 
                     description="Port number on which Milvus server is listening")
    user: str = Field("", env="MILVUS_USER", 
                     description="Username for authentication (if enabled on server)")
    password: str = Field("", env="MILVUS_PASSWORD", 
                         description="Password for authentication (if enabled on server)")
    secure: bool = Field(False, env="MILVUS_SECURE", 
                        description="Whether to use TLS/SSL for secure connection")
    timeout: int = Field(60, env="MILVUS_TIMEOUT", 
                        description="Connection timeout in seconds")
    connection_pool_size: int = Field(10, env="MILVUS_CONNECTION_POOL_SIZE", 
                                     description="Maximum number of connections to maintain in the pool")
    retry_count: int = Field(3, env="MILVUS_RETRY_COUNT", 
                            description="Number of times to retry failed operations")
    retry_interval: float = Field(1.0, env="MILVUS_RETRY_INTERVAL", 
                                 description="Time in seconds to wait between retry attempts")
    max_requests_per_second: int = Field(1000, env="MILVUS_MAX_REQUESTS_PER_SECOND",
                                        description="Maximum requests per second (rate limiting, 0=disabled)")
    rate_limiter_burst_multiplier: float = Field(2.0, env="MILVUS_RATE_LIMITER_BURST_MULTIPLIER",
                                                 description="Burst capacity multiplier for rate limiter")
    enable_retry_budget: bool = Field(True, env="MILVUS_ENABLE_RETRY_BUDGET",
                                     description="Enable retry budget to prevent retry storms")
    retry_budget_min_success_rate: float = Field(0.8, env="MILVUS_RETRY_BUDGET_MIN_SUCCESS_RATE",
                                                description="Minimum success rate (0.0-1.0) to allow retries")
    retry_budget_window_seconds: int = Field(10, env="MILVUS_RETRY_BUDGET_WINDOW_SECONDS",
                                            description="Time window for retry budget calculation")

    class Config:
        env_prefix = ""
        case_sensitive = False


class CollectionSettings(BaseSettings):
    """
    Collection settings for Milvus collections configuration.
    
    These settings control the behavior of Milvus collections, including:
    - ID generation strategy
    - Consistency guarantees for distributed operations
    - Schema versioning for collection evolution
    """
    auto_id: bool = Field(True, env="MILVUS_AUTO_ID",
                         description="Whether Milvus should automatically generate primary key IDs (true) or use provided IDs (false)")
    consistency_level: ConsistencyLevel = Field(ConsistencyLevel.STRONG, env="MILVUS_CONSISTENCY_LEVEL",
                                              description="Consistency level for read/write operations, balancing consistency vs performance")
    schema_version: str = Field("1", env="MILVUS_SCHEMA_VERSION",
                              description="Version identifier for the collection schema, useful for tracking schema evolution")

    class Config:
        env_prefix = ""
        case_sensitive = False
        use_enum_values = True


class HNSWIndexParams(BaseSettings):
    """
    HNSW (Hierarchical Navigable Small World) index parameters.
    
    HNSW is a graph-based indexing algorithm that offers excellent performance-accuracy trade-offs:
    - Creates a multi-layered graph structure for efficient navigation
    - Provides logarithmic search complexity
    - Supports dynamic data insertion without full rebuilds
    - Generally offers better recall than IVF-based indexes at the same speed
    """
    M: int = Field(16, env="MILVUS_HNSW_M",
                 description="Number of bi-directional links created for each new element (higher = better recall but more memory)")
    efConstruction: int = Field(200, env="MILVUS_HNSW_EF_CONSTRUCTION",
                              description="Size of the dynamic candidate list during index construction (higher = better quality but slower builds)")


class IVFIndexParams(BaseSettings):
    """
    IVF (Inverted File) index parameters.
    
    IVF is a clustering-based indexing algorithm:
    - Divides the vector space into clusters (nlist)
    - During search, only the closest clusters are examined (nprobe)
    - Offers good balance between memory usage, build speed, and search performance
    - Suitable for large-scale datasets where some accuracy can be traded for speed
    """
    nlist: int = Field(1024, env="MILVUS_IVF_NLIST",
                      description="Number of clusters to divide the vector space into (higher = better recall but slower builds and more memory)")


class HNSWSearchParams(BaseSettings):
    """
    HNSW search parameters for controlling search behavior at query time.
    
    These parameters control the trade-off between search speed and accuracy:
    - Can be adjusted at query time without rebuilding the index
    - Allow fine-tuning performance based on specific query requirements
    - Critical for optimizing search latency vs. recall rate
    """
    ef: int = Field(64, env="MILVUS_HNSW_EF",
                   description="Size of the dynamic candidate list during search (higher = better recall but slower search)")


class IVFSearchParams(BaseSettings):
    """
    IVF search parameters for controlling search behavior at query time.
    
    These parameters determine which and how many clusters are searched:
    - Can be adjusted at query time without rebuilding the index
    - Directly controls the trade-off between search speed and recall
    - One of the most important parameters for tuning IVF-based search performance
    """
    nprobe: int = Field(16, env="MILVUS_IVF_NPROBE",
                       description="Number of clusters to search during query (higher = better recall but slower search)")


class IndexSettings(BaseSettings):
    """
    Index settings for different index types supported by Milvus.
    
    This class consolidates settings for all index types in one place:
    - Provides consistent configuration interface for all index types
    - Allows selecting appropriate index type based on data characteristics
    - Configures metric types and parameters for each index
    - Enables optimization of index building for different use cases
    """
    # HNSW index settings
    hnsw_index_type: IndexType = Field(IndexType.HNSW, env="MILVUS_HNSW_INDEX_TYPE",
                                      description="Index type for HNSW-based indexes")
    hnsw_metric_type: MetricType = Field(MetricType.COSINE, env="MILVUS_HNSW_METRIC_TYPE",
                                        description="Distance metric for HNSW index (COSINE recommended for semantic search)")
    hnsw_params: HNSWIndexParams = Field(default_factory=HNSWIndexParams,
                                        description="HNSW-specific parameters for index building")
    
    # IVF index settings
    ivf_index_type: IndexType = Field(IndexType.IVF_FLAT, env="MILVUS_IVF_INDEX_TYPE",
                                     description="Index type for IVF-based indexes")
    ivf_metric_type: MetricType = Field(MetricType.COSINE, env="MILVUS_IVF_METRIC_TYPE",
                                       description="Distance metric for IVF index")
    ivf_params: IVFIndexParams = Field(default_factory=IVFIndexParams,
                                      description="IVF-specific parameters for index building")
    
    # FLAT index settings (exact search)
    flat_index_type: IndexType = Field(IndexType.FLAT, env="MILVUS_FLAT_INDEX_TYPE",
                                      description="Index type for exact (brute force) search")
    flat_metric_type: MetricType = Field(MetricType.COSINE, env="MILVUS_FLAT_METRIC_TYPE",
                                        description="Distance metric for FLAT index")
    
    # Sparse vector index settings
    sparse_index_type: IndexType = Field(IndexType.SPARSE_INVERTED_INDEX, env="MILVUS_SPARSE_INDEX_TYPE",
                                        description="Index type for sparse vectors (used in hybrid search)")
    sparse_metric_type: MetricType = Field(MetricType.IP, env="MILVUS_SPARSE_METRIC_TYPE",
                                          description="Distance metric for sparse vector index (IP recommended)")

    class Config:
        env_prefix = ""
        case_sensitive = False
        use_enum_values = True


class SearchSettings(BaseSettings):
    """
    Search settings for controlling query behavior across different index types.
    
    These settings determine how vector similarity searches are performed:
    - Configures search parameters for each index type
    - Controls the balance between search speed and accuracy
    - Enables fine-tuning of hybrid search weights
    - Can be adjusted at query time without rebuilding indexes
    """
    # HNSW search settings
    hnsw_metric_type: MetricType = Field(MetricType.COSINE, env="MILVUS_HNSW_SEARCH_METRIC_TYPE",
                                        description="Distance metric for HNSW search (should match index metric type)")
    hnsw_params: HNSWSearchParams = Field(default_factory=HNSWSearchParams,
                                         description="HNSW-specific parameters for search optimization")
    
    # IVF search settings
    ivf_metric_type: MetricType = Field(MetricType.COSINE, env="MILVUS_IVF_SEARCH_METRIC_TYPE",
                                       description="Distance metric for IVF search (should match index metric type)")
    ivf_params: IVFSearchParams = Field(default_factory=IVFSearchParams,
                                       description="IVF-specific parameters for search optimization")
    
    # FLAT search settings
    flat_metric_type: MetricType = Field(MetricType.COSINE, env="MILVUS_FLAT_SEARCH_METRIC_TYPE",
                                        description="Distance metric for FLAT search (exact search)")
    
    # Sparse vector search settings
    sparse_metric_type: MetricType = Field(MetricType.IP, env="MILVUS_SPARSE_SEARCH_METRIC_TYPE",
                                          description="Distance metric for sparse vector search")
    
    # Hybrid search weights
    hybrid_sparse_weight: float = Field(0.3, env="MILVUS_HYBRID_SPARSE_WEIGHT",
                                       description="Weight for sparse vector results in hybrid search (0.0-1.0)")
    hybrid_dense_weight: float = Field(0.7, env="MILVUS_HYBRID_DENSE_WEIGHT",
                                      description="Weight for dense vector results in hybrid search (0.0-1.0)")

    class Config:
        env_prefix = ""
        case_sensitive = False
        use_enum_values = True


class InsertionSettings(BaseSettings):
    """
    Insertion settings for controlling data ingestion into Milvus.
    
    These settings determine how data is inserted into Milvus collections:
    - Controls batch processing behavior for optimized throughput
    - Configures validation checks to ensure data integrity
    - Manages memory flushing behavior for durability
    - Sets timeouts to handle large insertion operations
    """
    batch_size: int = Field(100, env="MILVUS_BATCH_SIZE",
                           description="Number of vectors to insert in a single batch (higher = better throughput, more memory)")
    validate_data: bool = Field(True, env="MILVUS_VALIDATE_DATA",
                               description="Whether to validate vector dimensions and data types before insertion")
    auto_flush: bool = Field(True, env="MILVUS_AUTO_FLUSH",
                            description="Whether to automatically flush data to disk after insertion")
    timeout: int = Field(60, env="MILVUS_INSERTION_TIMEOUT",
                        description="Timeout in seconds for insertion operations")

    class Config:
        env_prefix = ""
        case_sensitive = False


class MonitoringSettings(BaseSettings):
    """
    Monitoring settings for tracking Milvus operations and performance.
    
    These settings configure how the system monitors and reports on Milvus operations:
    - Controls metrics collection for performance analysis
    - Configures logging verbosity for troubleshooting
    - Enables performance tracking for optimization
    - Sets intervals for metrics collection to balance overhead
    """
    enable_metrics: bool = Field(True, env="MILVUS_ENABLE_METRICS",
                                description="Whether to collect performance metrics for Milvus operations")
    log_level: str = Field("INFO", env="MILVUS_LOG_LEVEL",
                          description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    performance_tracking: bool = Field(True, env="MILVUS_PERFORMANCE_TRACKING",
                                      description="Whether to track detailed performance metrics (latency, throughput, etc.)")
    metrics_interval: int = Field(60, env="MILVUS_METRICS_INTERVAL",
                                 description="Interval in seconds between metrics collection points")

    class Config:
        env_prefix = ""
        case_sensitive = False


class BackupSettings(BaseSettings):
    """
    Backup settings for Milvus data protection and recovery.
    
    These settings configure how Milvus data is backed up and retained:
    - Specifies storage location for backup files
    - Controls compression to optimize storage space
    - Manages retention policy for backup history
    - Enables disaster recovery and point-in-time restore capabilities
    """
    backup_path: str = Field("./backups", env="MILVUS_BACKUP_PATH",
                            description="Directory path where backup files will be stored")
    compression: bool = Field(True, env="MILVUS_BACKUP_COMPRESSION",
                             description="Whether to compress backup files to save storage space")
    retention_days: int = Field(30, env="MILVUS_BACKUP_RETENTION_DAYS",
                               description="Number of days to retain backup files before automatic deletion")

    class Config:
        env_prefix = ""
        case_sensitive = False


class MilvusSettings(BaseSettings):
    """
    Main settings class for Milvus operations that consolidates all configuration categories.
    
    This class serves as the central configuration hub for the entire Milvus_Ops package:
    - Provides a unified interface for all Milvus-related settings
    - Supports loading from YAML files via YamlModelMixin
    - Enables environment variable overrides for all nested settings
    - Organizes settings into logical categories for better maintainability
    
    Usage:
        # Load from environment variables and defaults
        settings = MilvusSettings()
        
        # Load from YAML file
        settings = MilvusSettings.from_yaml('config.yaml')
        
        # Access nested settings
        host = settings.connection.host
        batch_size = settings.insertion.batch_size
    """
    connection: ConnectionSettings = Field(default_factory=ConnectionSettings,
                                         description="Connection settings for Milvus server")
    collection: CollectionSettings = Field(default_factory=CollectionSettings,
                                         description="Collection configuration settings")
    index: IndexSettings = Field(default_factory=IndexSettings,
                               description="Index configuration for different index types")
    search: SearchSettings = Field(default_factory=SearchSettings,
                                 description="Search parameters and configuration")
    insertion: InsertionSettings = Field(default_factory=InsertionSettings,
                                       description="Data insertion settings and behavior")
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings,
                                         description="Monitoring and metrics collection settings")
    backup: BackupSettings = Field(default_factory=BackupSettings,
                                 description="Backup and recovery configuration")

    class Config:
        env_prefix = ""
        case_sensitive = False
        env_nested_delimiter = "__"

    @classmethod
    def from_yaml(cls, yaml_file: Union[str, Path]) -> "MilvusSettings":
        """Load settings from YAML file"""
        import yaml
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


def load_settings(config_path: Optional[str] = None) -> MilvusSettings:
    """
    Load settings from file and/or environment variables.
    
    This function provides a convenient way to load settings from different sources:
    - If config_path is provided and exists, loads settings from the YAML file
    - Otherwise, creates a new settings instance with values from environment variables
    - The resulting settings object contains all configuration needed for Milvus operations
    
    Args:
        config_path: Path to YAML configuration file. If None or file doesn't exist,
                    falls back to environment variables and default values.
            
    Returns:
        MilvusSettings object with loaded configuration
        
    Example:
        # Load from specific config file
        settings = load_settings("/path/to/config.yaml")
        
        # Load from environment variables and defaults
        settings = load_settings()
    """
    if config_path and os.path.exists(config_path):
        return MilvusSettings.from_yaml(config_path)
    return MilvusSettings()
