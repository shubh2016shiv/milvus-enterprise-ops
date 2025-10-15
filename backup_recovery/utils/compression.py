"""
Compression Utilities

Provides data compression and decompression for backup operations with
support for multiple compression algorithms and configurable compression levels.
"""

import gzip
import logging
from pathlib import Path
from typing import Union, Optional
import shutil

logger = logging.getLogger(__name__)

# Try to import zstandard for better compression
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    logger.info("zstandard not available, falling back to gzip compression")


class CompressionHandler:
    """
    Handle compression and decompression of backup data.
    
    This class provides methods for compressing and decompressing files and
    data, with support for multiple compression algorithms. It automatically
    falls back to gzip if zstandard is not available.
    
    Supported compression methods:
        - gzip: Widely supported, good compression
        - zstandard: Faster compression/decompression, better ratios (if available)
    
    Example:
        ```python
        handler = CompressionHandler(compression_level=6, prefer_zstd=True)
        
        # Compress a file
        compressed_path = await handler.compress_file(
            "/backups/data.parquet",
            "/backups/data.parquet.gz"
        )
        
        # Decompress a file
        decompressed_path = await handler.decompress_file(
            "/backups/data.parquet.gz",
            "/restore/data.parquet"
        )
        ```
    """
    
    # Buffer size for streaming operations (1 MB)
    BUFFER_SIZE = 1024 * 1024
    
    def __init__(
        self,
        compression_level: int = 6,
        prefer_zstd: bool = True
    ):
        """
        Initialize compression handler.
        
        Args:
            compression_level: Compression level from 1 (fastest) to 9 (best)
            prefer_zstd: Prefer zstandard over gzip if available
        """
        if not 1 <= compression_level <= 9:
            raise ValueError("compression_level must be between 1 and 9")
        
        self.compression_level = compression_level
        self.use_zstd = prefer_zstd and ZSTD_AVAILABLE
        
        if self.use_zstd:
            logger.debug(f"Using zstandard compression at level {compression_level}")
        else:
            logger.debug(f"Using gzip compression at level {compression_level}")
    
    async def compress_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Compress a file.
        
        If output_path is not provided, adds appropriate extension to input path.
        Uses streaming to handle large files efficiently.
        
        Args:
            input_path: Path to file to compress
            output_path: Path for compressed output (optional)
        
        Returns:
            Path to compressed file
        
        Raises:
            FileNotFoundError: If input file doesn't exist
            IOError: If compression fails
        
        Example:
            ```python
            compressed = await handler.compress_file("/backups/data.parquet")
            # Creates /backups/data.parquet.gz or .zst
            ```
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Determine output path
        if output_path is None:
            extension = ".zst" if self.use_zstd else ".gz"
            output_path = input_path.with_suffix(input_path.suffix + extension)
        else:
            output_path = Path(output_path)
        
        try:
            if self.use_zstd:
                await self._compress_file_zstd(input_path, output_path)
            else:
                await self._compress_file_gzip(input_path, output_path)
            
            # Log compression ratio
            original_size = input_path.stat().st_size
            compressed_size = output_path.stat().st_size
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            logger.info(
                f"Compressed {input_path.name}: "
                f"{original_size / (1024**2):.2f} MB → {compressed_size / (1024**2):.2f} MB "
                f"({ratio:.1f}% reduction)"
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to compress {input_path}: {e}")
            # Clean up partial output
            if output_path.exists():
                output_path.unlink()
            raise IOError(f"Compression failed: {e}")
    
    async def _compress_file_gzip(self, input_path: Path, output_path: Path) -> None:
        """Compress file using gzip."""
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb', compresslevel=self.compression_level) as f_out:
                while True:
                    chunk = f_in.read(self.BUFFER_SIZE)
                    if not chunk:
                        break
                    f_out.write(chunk)
    
    async def _compress_file_zstd(self, input_path: Path, output_path: Path) -> None:
        """Compress file using zstandard."""
        cctx = zstd.ZstdCompressor(level=self.compression_level)
        
        with open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                cctx.copy_stream(f_in, f_out, size=input_path.stat().st_size)
    
    async def decompress_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Decompress a file.
        
        Automatically detects compression format from file extension.
        If output_path is not provided, removes compression extension.
        
        Args:
            input_path: Path to compressed file
            output_path: Path for decompressed output (optional)
        
        Returns:
            Path to decompressed file
        
        Raises:
            FileNotFoundError: If input file doesn't exist
            IOError: If decompression fails
        
        Example:
            ```python
            decompressed = await handler.decompress_file("/backups/data.parquet.gz")
            # Creates /backups/data.parquet
            ```
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Determine output path
        if output_path is None:
            if input_path.suffix in ['.gz', '.zst']:
                output_path = input_path.with_suffix('')
            else:
                output_path = input_path.with_suffix('.decompressed')
        else:
            output_path = Path(output_path)
        
        try:
            # Detect compression format
            if input_path.suffix == '.zst':
                await self._decompress_file_zstd(input_path, output_path)
            elif input_path.suffix == '.gz':
                await self._decompress_file_gzip(input_path, output_path)
            else:
                raise ValueError(f"Unknown compression format: {input_path.suffix}")
            
            logger.info(f"Decompressed {input_path.name} to {output_path.name}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to decompress {input_path}: {e}")
            # Clean up partial output
            if output_path.exists():
                output_path.unlink()
            raise IOError(f"Decompression failed: {e}")
    
    async def _decompress_file_gzip(self, input_path: Path, output_path: Path) -> None:
        """Decompress file using gzip."""
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                while True:
                    chunk = f_in.read(self.BUFFER_SIZE)
                    if not chunk:
                        break
                    f_out.write(chunk)
    
    async def _decompress_file_zstd(self, input_path: Path, output_path: Path) -> None:
        """Decompress file using zstandard."""
        if not ZSTD_AVAILABLE:
            raise RuntimeError("zstandard not available for decompression")
        
        dctx = zstd.ZstdDecompressor()
        
        with open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                dctx.copy_stream(f_in, f_out)
    
    def compress_data(self, data: bytes) -> bytes:
        """
        Compress in-memory data.
        
        Suitable for small data like metadata or configuration files.
        
        Args:
            data: Bytes to compress
        
        Returns:
            Compressed bytes
        
        Example:
            ```python
            data = b"some binary data"
            compressed = handler.compress_data(data)
            ```
        """
        try:
            if self.use_zstd:
                cctx = zstd.ZstdCompressor(level=self.compression_level)
                compressed = cctx.compress(data)
            else:
                compressed = gzip.compress(data, compresslevel=self.compression_level)
            
            ratio = (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0
            logger.debug(
                f"Compressed {len(data)} bytes to {len(compressed)} bytes ({ratio:.1f}% reduction)"
            )
            
            return compressed
            
        except Exception as e:
            logger.error(f"Failed to compress data: {e}")
            raise ValueError(f"Data compression failed: {e}")
    
    def decompress_data(self, data: bytes, is_zstd: Optional[bool] = None) -> bytes:
        """
        Decompress in-memory data.
        
        Args:
            data: Compressed bytes
            is_zstd: Whether data is zstd compressed (auto-detects if None)
        
        Returns:
            Decompressed bytes
        
        Example:
            ```python
            compressed = b"..."
            decompressed = handler.decompress_data(compressed)
            ```
        """
        try:
            # Auto-detect format if not specified
            if is_zstd is None:
                # zstd has a 4-byte magic number: 0x28, 0xB5, 0x2F, 0xFD
                is_zstd = (
                    len(data) >= 4 and
                    data[0:4] == b'\x28\xB5\x2F\xFD'
                )
            
            if is_zstd:
                if not ZSTD_AVAILABLE:
                    raise RuntimeError("zstandard not available for decompression")
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.decompress(data)
            else:
                decompressed = gzip.decompress(data)
            
            logger.debug(f"Decompressed {len(data)} bytes to {len(decompressed)} bytes")
            return decompressed
            
        except Exception as e:
            logger.error(f"Failed to decompress data: {e}")
            raise ValueError(f"Data decompression failed: {e}")
    
    @staticmethod
    def get_compression_extension(use_zstd: bool = False) -> str:
        """
        Get the appropriate file extension for compression format.
        
        Args:
            use_zstd: Whether to use zstandard
        
        Returns:
            File extension ('.gz' or '.zst')
        """
        return '.zst' if use_zstd and ZSTD_AVAILABLE else '.gz'
    
    @staticmethod
    def is_compressed(file_path: Union[str, Path]) -> bool:
        """
        Check if a file appears to be compressed based on extension.
        
        Args:
            file_path: Path to check
        
        Returns:
            True if file has compression extension
        """
        return Path(file_path).suffix in ['.gz', '.zst', '.bz2', '.xz']
    
    def get_estimated_compressed_size(self, original_size: int) -> int:
        """
        Estimate compressed size based on typical compression ratios.
        
        This is a rough estimate and actual results may vary significantly
        based on data characteristics.
        
        Args:
            original_size: Original file size in bytes
        
        Returns:
            Estimated compressed size in bytes
        """
        # Typical compression ratios (conservative estimates)
        if self.use_zstd:
            # zstd typically achieves 2-4x compression
            ratio = 0.4  # Assume 40% of original size
        else:
            # gzip typically achieves 2-3x compression
            ratio = 0.5  # Assume 50% of original size
        
        return int(original_size * ratio)

