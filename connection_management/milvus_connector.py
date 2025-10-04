"""
Milvus Connector - Simplified Connection Interface

This module provides a high-level, simplified interface for connecting to Milvus
while leveraging all the underlying resiliency and fault tolerance patterns of the
connection management module. It is designed for ease of use in production
environments where robust connection handling is critical.
"""
import os
import sys
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import uuid
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from config import MilvusSettings, load_settings
from connection_management.connection_manager import ConnectionManager
from connection_management.connection_exceptions import (
    ConnectionError,
    ServerUnavailableError
)

logger = logging.getLogger(__name__)


class ConnectionStatus(str, Enum):
    """
    Defines the possible states of a Milvus connection attempt.
    
    This enum provides clear, standardized feedback on the connection status,
    enabling applications to implement appropriate logic for each state.
    """
    SUCCESS = "success"
    FAILURE = "failure"
    UNAVAILABLE = "unavailable"
    PENDING = "pending"


@dataclass
class ConnectionFeedback:
    """
    Provides detailed feedback on the Milvus connection attempt.
    
    This data class encapsulates all relevant information about a connection
    attempt, including a unique correlation ID for tracing, the connection status,
    and a descriptive message for logging and debugging.
    """
    milvus_connection_id: str
    status: ConnectionStatus
    message: str
    connection_manager: Optional[ConnectionManager] = None


class MilvusConnector:
    """
    High-level connector for establishing robust Milvus connections.
    
    This class abstracts away the complexities of connection pooling, circuit
    breakers, and retry logic, providing a simple `connect` method that
    returns a clear status and a unique correlation ID.
    
    The connector is designed for production use cases where millions of users
    require reliable and resilient connections to Milvus.
    """
    
    def __init__(
        self, 
        config: Optional[MilvusSettings] = None, 
        enable_circuit_breaker: bool = True
    ):
        """
        Initialize the Milvus connector.
        
        Args:
            config: MilvusSettings object. If None, default settings are loaded.
            enable_circuit_breaker: Whether to enable circuit breaker protection.
                                   Recommended for production environments.
        """

        self.config = config or load_settings()
        self.enable_circuit_breaker = enable_circuit_breaker
        self._connection_manager: Optional[ConnectionManager] = None
        
        logger.info(
            f"MilvusConnector initialized with circuit breaker "
            f"{'enabled' if enable_circuit_breaker else 'disabled'}"
        )
    
    def establish_connection(self) -> ConnectionFeedback:
        """
        Establish a resilient connection to Milvus.
        
        This method leverages the underlying ConnectionManager to create a
        robust connection, handle failures gracefully, and provide clear
        feedback on the connection status.
        
        The method returns a ConnectionFeedback object with:
        - A unique `milvus_connection_id` for tracing
        - The connection `status` (SUCCESS, FAILURE, UNAVAILABLE)
        - A descriptive `message` for logging and debugging
        
        Returns:
            ConnectionFeedback: Detailed feedback on the connection attempt.
        """
        milvus_connection_id = f"milvus-conn-{uuid.uuid4()}"
        logger.info(
            f"[{milvus_connection_id}] Attempting to establish Milvus connection..."
        )
        
        try:
            # Initialize the connection manager, which creates the pool
            self._connection_manager = ConnectionManager(
                self.config, self.enable_circuit_breaker
            )
            
            # Check server status to confirm connectivity
            is_available = self._connection_manager.check_server_status()
            
            if is_available:
                logger.info(
                    f"[{milvus_connection_id}] Milvus connection established successfully."
                )
                return ConnectionFeedback(
                    milvus_connection_id=milvus_connection_id,
                    status=ConnectionStatus.SUCCESS,
                    message="Milvus connection established successfully.",
                    connection_manager=self._connection_manager
                )
            else:
                # This could happen if circuit breaker is open or all retries fail
                logger.warning(
                    f"[{milvus_connection_id}] Milvus server is unavailable. "
                    "This may be due to an open circuit breaker or persistent connection issues."
                )
                return ConnectionFeedback(
                    milvus_connection_id=milvus_connection_id,
                    status=ConnectionStatus.UNAVAILABLE,
                    message="Milvus server is unavailable.",
                    connection_manager=self._connection_manager
                )
                
        except (ConnectionError, ServerUnavailableError) as e:
            logger.error(
                f"[{milvus_connection_id}] Failed to establish Milvus connection: {e}",
                exc_info=True
            )
            return ConnectionFeedback(
                milvus_connection_id=milvus_connection_id,
                status=ConnectionStatus.FAILURE,
                message=f"Failed to establish Milvus connection: {e}",
                connection_manager=self._connection_manager
            )
        
        except Exception as e:
            logger.critical(
                f"[{milvus_connection_id}] An unexpected error occurred during "
                f"Milvus connection establishment: {e}",
                exc_info=True
            )
            return ConnectionFeedback(
                milvus_connection_id=milvus_connection_id,
                status=ConnectionStatus.FAILURE,
                message=f"An unexpected error occurred: {e}",
                connection_manager=self._connection_manager
            )
    
    def get_connection_manager(self) -> Optional[ConnectionManager]:
        """
        Get the underlying ConnectionManager instance.
        
        Returns:
            Optional[ConnectionManager]: The ConnectionManager instance, or None
                                        if connection has not been established.
        """
        return self._connection_manager
    
    def close_connection(self):
        """
        Close the connection and release all resources.
        """
        if self._connection_manager:
            self._connection_manager.close()
            logger.info("Milvus connection manager closed.")
        else:
            logger.warning("No active connection manager to close.")
    
    def __enter__(self):
        """Enter context manager, establish connection."""
        feedback = self.establish_connection()
        if feedback.status != ConnectionStatus.SUCCESS:
            raise ConnectionError(
                f"Failed to establish connection: {feedback.message}"
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, close connection."""
        self.close_connection()


if __name__ == "__main__":
    # Import logging configuration
    import logging
    # Set up root logger to show all logs
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Make sure our module's logger is also set to DEBUG
    logger.setLevel(logging.DEBUG)
    print("Starting Milvus connector test...")
    print("Import path fixed successfully!")
    
    try:
        print("Loading configuration from default settings...")
        from config.settings import load_settings
        config = load_settings()
        print(f"Configuration loaded. Host: {config.connection.host}, Port: {config.connection.port}")
        
        # Create a connector instance
        print("Creating MilvusConnector instance...")
        connector = MilvusConnector(config)
        
        # Test connection establishment
        print("\nAttempting to connect to Milvus server...")
        connection_feedback = connector.establish_connection()
    except Exception as e:
        print(f"Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Check connection status
    print(f"Connection status: {connection_feedback.status}")
    print(f"Connection message: {connection_feedback.message}")
    print(f"Connection ID: {connection_feedback.milvus_connection_id}")
    
    # If connection was successful, try to execute a simple operation
    if connection_feedback.status == ConnectionStatus.SUCCESS:
        print("\nConnection successful! Testing a simple operation...")
        
        # Get the connection manager
        conn_manager = connection_feedback.connection_manager
        
        # Define a simple operation to list collections
        def list_collections_operation(conn_alias):
            """Simple operation to list all collections in Milvus."""
            from pymilvus import utility
            try:
                collections = utility.list_collections(using=conn_alias)
                return collections
            except Exception as e:
                print(f"Error listing collections: {e}")
                return []
        
        try:
            # Execute the operation with automatic retry and circuit breaker protection
            collections = conn_manager.execute_operation(list_collections_operation)
            print(f"Available collections: {collections}")
            
            # Get circuit breaker metrics if available
            metrics = conn_manager.get_circuit_breaker_metrics()
            if metrics:
                print("\nCircuit Breaker Metrics:")
                print(f"  State: {metrics['state']}")
                print(f"  Total Requests: {metrics['counters']['total_requests']}")
                print(f"  Success Rate: {metrics['current_state']['success_rate_percent']}%")
        except Exception as e:
            print(f"Operation failed: {e}")
    else:
        print("\nFailed to connect to Milvus server. Check your configuration and server status.")
    
    # Clean up resources
    print("\nClosing connection...")
    connector.close_connection()
    print("Connection closed.")