"""
Checksum Calculation Utilities

Provides robust checksum calculation and verification for backup integrity.
Supports multiple hash algorithms with streaming for large files.
"""

import hashlib
import logging
from pathlib import Path
from typing import Union, BinaryIO, Optional
from ..models.entities import ChecksumAlgorithm

logger = logging.getLogger(__name__)


class ChecksumCalculator:
    """
    Calculate and verify checksums for backup data integrity.
    
    This class provides methods for calculating checksums of files and data
    using various algorithms, with support for streaming large files to
    avoid loading them entirely into memory.
    
    Supported algorithms:
        - SHA256: Secure, recommended for production
        - MD5: Fast but less secure, suitable for non-critical use
        - BLAKE2B: Fast and secure, modern alternative
    
    Example:
        ```python
        calculator = ChecksumCalculator(ChecksumAlgorithm.SHA256)
        
        # Calculate checksum for a file
        checksum = await calculator.calculate_file_checksum("/path/to/file")
        
        # Verify file against known checksum
        is_valid = await calculator.verify_file_checksum(
            "/path/to/file",
            expected_checksum=checksum
        )
        ```
    """
    
    # Buffer size for streaming file reads (1 MB)
    BUFFER_SIZE = 1024 * 1024
    
    def __init__(self, algorithm: ChecksumAlgorithm = ChecksumAlgorithm.SHA256):
        """
        Initialize checksum calculator with specified algorithm.
        
        Args:
            algorithm: Hash algorithm to use for checksum calculation
        """
        self.algorithm = algorithm
        logger.debug(f"ChecksumCalculator initialized with {algorithm.value}")
    
    def _get_hash_function(self):
        """
        Get the hash function for the configured algorithm.
        
        Returns:
            Hash function object
        
        Raises:
            ValueError: If algorithm is not supported
        """
        if self.algorithm == ChecksumAlgorithm.SHA256:
            return hashlib.sha256()
        elif self.algorithm == ChecksumAlgorithm.MD5:
            return hashlib.md5()
        elif self.algorithm == ChecksumAlgorithm.BLAKE2B:
            return hashlib.blake2b()
        else:
            raise ValueError(f"Unsupported checksum algorithm: {self.algorithm}")
    
    async def calculate_file_checksum(
        self,
        file_path: Union[str, Path],
        buffer_size: Optional[int] = None
    ) -> str:
        """
        Calculate checksum for a file using streaming to handle large files.
        
        This method reads the file in chunks to avoid loading the entire
        file into memory, making it suitable for very large backup files.
        
        Args:
            file_path: Path to the file
            buffer_size: Size of read buffer (uses default if None)
        
        Returns:
            Hexadecimal checksum string
        
        Raises:
            FileNotFoundError: If file does not exist
            IOError: If file cannot be read
        
        Example:
            ```python
            checksum = await calculator.calculate_file_checksum("/backups/data.parquet")
            print(f"File checksum: {checksum}")
            ```
        """
        file_path = Path(file_path)
        buffer_size = buffer_size or self.BUFFER_SIZE
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            hash_func = self._get_hash_function()
            
            with open(file_path, 'rb') as f:
                while True:
                    data = f.read(buffer_size)
                    if not data:
                        break
                    hash_func.update(data)
            
            checksum = hash_func.hexdigest()
            logger.debug(f"Calculated checksum for {file_path.name}: {checksum[:16]}...")
            return checksum
            
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            raise IOError(f"Error reading file for checksum: {e}")
    
    def calculate_data_checksum(self, data: bytes) -> str:
        """
        Calculate checksum for in-memory data.
        
        This method is suitable for small data that is already in memory,
        such as metadata or configuration files.
        
        Args:
            data: Bytes to calculate checksum for
        
        Returns:
            Hexadecimal checksum string
        
        Example:
            ```python
            data = b"some binary data"
            checksum = calculator.calculate_data_checksum(data)
            ```
        """
        try:
            hash_func = self._get_hash_function()
            hash_func.update(data)
            checksum = hash_func.hexdigest()
            logger.debug(f"Calculated checksum for {len(data)} bytes: {checksum[:16]}...")
            return checksum
        except Exception as e:
            logger.error(f"Failed to calculate checksum for data: {e}")
            raise ValueError(f"Error calculating checksum: {e}")
    
    async def verify_file_checksum(
        self,
        file_path: Union[str, Path],
        expected_checksum: str,
        buffer_size: Optional[int] = None
    ) -> bool:
        """
        Verify a file's checksum against an expected value.
        
        Args:
            file_path: Path to the file to verify
            expected_checksum: Expected checksum value
            buffer_size: Size of read buffer (uses default if None)
        
        Returns:
            True if checksums match, False otherwise
        
        Example:
            ```python
            is_valid = await calculator.verify_file_checksum(
                "/backups/data.parquet",
                expected_checksum="abc123..."
            )
            
            if not is_valid:
                print("File integrity check failed!")
            ```
        """
        try:
            actual_checksum = await self.calculate_file_checksum(file_path, buffer_size)
            matches = actual_checksum.lower() == expected_checksum.lower()
            
            if matches:
                logger.debug(f"Checksum verification passed for {Path(file_path).name}")
            else:
                logger.warning(
                    f"Checksum mismatch for {Path(file_path).name}: "
                    f"expected {expected_checksum[:16]}..., got {actual_checksum[:16]}..."
                )
            
            return matches
            
        except Exception as e:
            logger.error(f"Checksum verification failed for {file_path}: {e}")
            return False
    
    def verify_data_checksum(self, data: bytes, expected_checksum: str) -> bool:
        """
        Verify data's checksum against an expected value.
        
        Args:
            data: Bytes to verify
            expected_checksum: Expected checksum value
        
        Returns:
            True if checksums match, False otherwise
        
        Example:
            ```python
            data = b"some binary data"
            is_valid = calculator.verify_data_checksum(data, expected_checksum)
            ```
        """
        try:
            actual_checksum = self.calculate_data_checksum(data)
            matches = actual_checksum.lower() == expected_checksum.lower()
            
            if not matches:
                logger.warning(
                    f"Data checksum mismatch: "
                    f"expected {expected_checksum[:16]}..., got {actual_checksum[:16]}..."
                )
            
            return matches
            
        except Exception as e:
            logger.error(f"Data checksum verification failed: {e}")
            return False
    
    async def calculate_directory_checksum(
        self,
        directory_path: Union[str, Path],
        include_hidden: bool = False
    ) -> str:
        """
        Calculate a combined checksum for all files in a directory.
        
        This creates a single checksum representing all files in the directory,
        useful for verifying entire backup directories.
        
        Args:
            directory_path: Path to the directory
            include_hidden: Whether to include hidden files
        
        Returns:
            Combined checksum for all files
        
        Raises:
            NotADirectoryError: If path is not a directory
        
        Example:
            ```python
            checksum = await calculator.calculate_directory_checksum("/backups/collection_backup")
            ```
        """
        directory_path = Path(directory_path)
        
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        # Get all files sorted by name for consistent ordering
        files = sorted(
            [f for f in directory_path.rglob('*') if f.is_file()],
            key=lambda x: str(x.relative_to(directory_path))
        )
        
        # Filter hidden files if requested
        if not include_hidden:
            files = [f for f in files if not any(part.startswith('.') for part in f.parts)]
        
        # Calculate combined checksum
        hash_func = self._get_hash_function()
        
        for file_path in files:
            # Include relative path in hash for structure verification
            relative_path = str(file_path.relative_to(directory_path))
            hash_func.update(relative_path.encode('utf-8'))
            
            # Add file checksum
            file_checksum = await self.calculate_file_checksum(file_path)
            hash_func.update(file_checksum.encode('utf-8'))
        
        combined_checksum = hash_func.hexdigest()
        logger.info(f"Calculated directory checksum for {len(files)} files: {combined_checksum[:16]}...")
        return combined_checksum
    
    @staticmethod
    def generate_checksum_with_algorithm(
        data: bytes,
        algorithm: ChecksumAlgorithm
    ) -> str:
        """
        Static helper to generate checksum with specific algorithm.
        
        Convenience method for one-off checksum calculations without
        creating a calculator instance.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm to use
        
        Returns:
            Hexadecimal checksum string
        
        Example:
            ```python
            checksum = ChecksumCalculator.generate_checksum_with_algorithm(
                data=b"some data",
                algorithm=ChecksumAlgorithm.SHA256
            )
            ```
        """
        calculator = ChecksumCalculator(algorithm)
        return calculator.calculate_data_checksum(data)

