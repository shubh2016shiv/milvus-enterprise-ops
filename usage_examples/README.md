# Milvus Operations - Usage Examples

Comprehensive collection of practical examples demonstrating each feature of the Milvus_Ops project.

## 📋 Prerequisites

### 1. Running Milvus Instance

Ensure Milvus is running via Docker:

```bash
docker ps | grep milvus
```

If not running, start Milvus:

```bash
cd milvus_infra_management
docker-compose up -d
```

### 2. Python Environment

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Edit `usage_examples/config.yaml` if your Milvus instance uses non-default settings:

```yaml
milvus:
  connection:
    host: "localhost"
    port: "19530"
    # ... other settings
```

## 📁 Directory Structure

```
usage_examples/
├── 01_connection/          # Connection management examples
├── 02_collection_ops/      # Collection operations
├── 03_data_management/     # Data insertion, updates, deletion
├── 04_index_ops/          # Index creation and management
├── 05_search_ops/         # Search and retrieval
├── 06_backup_recovery/    # Backup and restore operations
├── config.yaml            # Configuration file
├── utils.py               # Common utilities
└── README.md              # This file
```

## 🚀 Quick Start

### Recommended Execution Sequence

Run examples in this order for a complete workflow:

```bash
# 1. Test Connection
cd usage_examples
python 01_connection/basic_connection.py

# 2. Create Collection
python 02_collection_ops/create_collection.py

# 3. Insert Data
python 03_data_management/insert_data.py

# 4. Create Index
python 04_index_ops/create_ivf_index.py

# 5. Search Data
python 05_search_ops/semantic_search.py

# 6. Create Backup
python 06_backup_recovery/create_backup.py
```

## 📖 Detailed Examples

### 1. Connection Management (01_connection/)

#### basic_connection.py
Tests basic connection to Milvus server.

```bash
python 01_connection/basic_connection.py
```

**What it demonstrates:**
- Connecting to Milvus
- Checking server status
- Retrieving connection info
- Proper connection cleanup

#### connection_pool_test.py
Tests connection pooling with concurrent operations.

```bash
python 01_connection/connection_pool_test.py
```

**What it demonstrates:**
- Connection pool management
- Concurrent operations
- Pool exhaustion handling
- Connection metrics

#### circuit_breaker_test.py
Demonstrates circuit breaker functionality for fault tolerance.

```bash
python 01_connection/circuit_breaker_test.py
```

**What it demonstrates:**
- Circuit breaker states (CLOSED, OPEN, HALF_OPEN)
- Failure detection
- Automatic recovery
- Fail-fast behavior

---

### 2. Collection Operations (02_collection_ops/)

#### create_collection.py
Creates a collection with a defined schema.

```bash
python 02_collection_ops/create_collection.py
```

**What it demonstrates:**
- Defining collection schema
- Setting field types and properties
- Creating vector fields
- Enabling dynamic fields

#### load_collection.py
Loads a collection into memory for search operations.

```bash
python 02_collection_ops/load_collection.py
```

**What it demonstrates:**
- Loading collections
- Monitoring load progress
- Checking load state
- Making collections search-ready

#### collection_stats.py
Retrieves and displays collection statistics.

```bash
python 02_collection_ops/collection_stats.py
```

**What it demonstrates:**
- Getting collection description
- Retrieving entity counts
- Viewing segment information
- Checking memory usage

#### drop_collection.py
Drops (deletes) a collection.

```bash
python 02_collection_ops/drop_collection.py
```

**What it demonstrates:**
- Safely dropping collections
- Verifying deletion
- Cleanup procedures

---

### 3. Data Management (03_data_management/)

#### insert_data.py
Inserts sample vectors and metadata.

```bash
python 03_data_management/insert_data.py
```

**What it demonstrates:**
- Preparing data for insertion
- Inserting entities
- Flushing data to disk
- Verifying insertion success

#### batch_insert.py
Efficiently inserts large amounts of data in batches.

```bash
python 03_data_management/batch_insert.py
```

**What it demonstrates:**
- Batch insertion for performance
- Progress monitoring
- Handling partial failures
- Throughput optimization

#### upsert_data.py
Updates existing entities or inserts new ones.

```bash
python 03_data_management/upsert_data.py
```

**What it demonstrates:**
- Upsert operations
- Updating vs inserting logic
- Primary key matching
- Verifying updates

#### delete_data.py
Deletes entities by IDs or filter expressions.

```bash
python 03_data_management/delete_data.py
```

**What it demonstrates:**
- Delete by IDs
- Delete by expression
- Filter syntax
- Verifying deletions

---

### 4. Index Operations (04_index_ops/)

#### create_ivf_index.py
Creates an IVF_FLAT index for efficient search.

```bash
python 04_index_ops/create_ivf_index.py
```

**What it demonstrates:**
- IVF_FLAT index creation
- Index parameter tuning (nlist)
- Monitoring index build
- Index verification

#### create_hnsw_index.py
Creates an HNSW index for high-performance search.

```bash
python 04_index_ops/create_hnsw_index.py
```

**What it demonstrates:**
- HNSW index creation
- Parameter tuning (M, efConstruction)
- Performance vs accuracy tradeoffs
- Index comparison

#### index_progress.py
Monitors index building progress in real-time.

```bash
python 04_index_ops/index_progress.py
```

**What it demonstrates:**
- Real-time progress monitoring
- Progress bar display
- Index build metrics
- Completion detection

#### drop_index.py
Drops an existing index from a collection.

```bash
python 04_index_ops/drop_index.py
```

**What it demonstrates:**
- Index removal
- Releasing collections
- Brute-force search fallback
- Index management

---

### 5. Search Operations (05_search_ops/)

#### semantic_search.py
Performs basic vector similarity search.

```bash
python 05_search_ops/semantic_search.py
```

**What it demonstrates:**
- Vector search basics
- Setting search parameters
- Top-K retrieval
- Batch queries

#### filtered_search.py
Searches with metadata filters and expressions.

```bash
python 05_search_ops/filtered_search.py
```

**What it demonstrates:**
- Combining vector search with filters
- Filter expression syntax
- Numeric and string filters
- AND/OR conditions

#### hybrid_search.py
Combines dense (vector) and sparse (keyword) retrieval.

```bash
python 05_search_ops/hybrid_search.py
```

**What it demonstrates:**
- Hybrid search strategy
- Dense + sparse fusion
- Reciprocal Rank Fusion (RRF)
- Result combination

#### reranking_search.py
Demonstrates result reranking for improved relevance.

```bash
python 05_search_ops/reranking_search.py
```

**What it demonstrates:**
- Two-stage retrieval
- Reranking strategies
- Relevance improvement
- Ranking comparison

---

### 6. Backup and Recovery (06_backup_recovery/)

#### create_backup.py
Creates a backup of a collection.

```bash
python 06_backup_recovery/create_backup.py
```

**What it demonstrates:**
- Backup creation
- Compression and checksums
- Backup metadata
- Storage management

#### restore_backup.py
Restores a collection from backup.

```bash
python 06_backup_recovery/restore_backup.py
```

**What it demonstrates:**
- Backup restoration
- Integrity verification
- Data recovery
- Restore validation

#### verify_backup.py
Verifies backup integrity and health.

```bash
python 06_backup_recovery/verify_backup.py
```

**What it demonstrates:**
- Checksum validation
- Metadata inspection
- Health assessment
- Maintenance recommendations

---

## 🔧 Common Utilities

### utils.py

Provides shared helper functions:

- `generate_random_vectors()` - Create test vectors
- `generate_test_data()` - Generate complete test datasets
- `print_section()` - Format output sections
- `print_step()` - Display step progress
- `print_success()` / `print_error()` / `print_warning()` - Status messages
- `print_info()` - Key-value information
- `cleanup_collection()` - Clean up test data

## 🎯 Use Cases

### Learning Workflow

1. **Beginners**: Start with connection examples, then collection operations
2. **Intermediate**: Focus on data management and search operations
3. **Advanced**: Explore index optimization and backup strategies

### Development Workflow

```bash
# Setup
python 02_collection_ops/create_collection.py

# Development
python 03_data_management/batch_insert.py
python 04_index_ops/create_hnsw_index.py
python 05_search_ops/semantic_search.py

# Cleanup
python 02_collection_ops/drop_collection.py
```

### Production Testing

```bash
# Test resilience
python 01_connection/circuit_breaker_test.py
python 01_connection/connection_pool_test.py

# Test data operations
python 03_data_management/batch_insert.py

# Test backup/recovery
python 06_backup_recovery/create_backup.py
python 06_backup_recovery/restore_backup.py
python 06_backup_recovery/verify_backup.py
```

## 🐛 Troubleshooting

### Connection Errors

```
Error: Failed to connect to Milvus server
```

**Solution:**
1. Verify Milvus is running: `docker ps | grep milvus`
2. Check port 19530 is accessible
3. Review `config.yaml` settings

### Collection Not Found

```
Error: Collection 'xxx' does not exist
```

**Solution:**
Run `create_collection.py` first to create the collection.

### Index Build Errors

```
Error: Cannot create index while collection is loaded
```

**Solution:**
Release the collection before creating an index:
```python
collection.release()
```

### Search Errors

```
Error: Collection not loaded
```

**Solution:**
Load the collection before searching:
```bash
python 02_collection_ops/load_collection.py
```

## 📊 Performance Tips

### Data Insertion
- Use batch sizes of 100-1000 for optimal throughput
- Monitor memory usage during large inserts
- Flush data periodically

### Index Creation
- Choose index type based on dataset size:
  - < 100K vectors: FLAT or IVF_FLAT
  - 100K - 1M vectors: IVF_FLAT or IVF_PQ
  - > 1M vectors: HNSW

### Search Optimization
- Load collections before searching
- Use appropriate Top-K values
- Apply filters to reduce search space
- Consider hybrid search for complex queries

## 🔐 Best Practices

1. **Always close connections** after operations
2. **Use try-except blocks** for error handling
3. **Verify operations** after execution
4. **Test with small datasets** before scaling
5. **Monitor resource usage** during operations
6. **Create regular backups** of important collections
7. **Test restore procedures** periodically

## 📝 Example Output

Typical output from running an example:

```
============================================================
 Create Collection Example
============================================================

[Step 1] Initialize Managers
✓ Managers initialized

[Step 2] Define Collection Schema
  • Vector dimension: 128
  • Fields: 5
✓ Schema defined

[Step 3] Create Collection
✓ Collection 'test_example_collection' created successfully

[Step 4] Verify Collection
✓ Collection exists and is ready

============================================================
 Example Completed
============================================================
```

## 🤝 Contributing

To add new examples:

1. Place scripts in appropriate subdirectory
2. Follow existing naming conventions
3. Include comprehensive comments
4. Add print statements for progress
5. Update this README

## 📚 Additional Resources

- [Milvus Documentation](https://milvus.io/docs)
- [PyMilvus API Reference](https://milvus.io/api-reference/pymilvus/v2.3.x/About.md)
- Project main README: `../README.md`

## ⚖️ License

Same as main project license.

---

**Note:** All examples use a test collection named `test_example_collection`. You can safely drop this collection after testing.

For questions or issues, please refer to the main project documentation or create an issue in the repository.


