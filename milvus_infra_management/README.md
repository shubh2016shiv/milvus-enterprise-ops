# Milvus Infrastructure Management

This directory contains tools for managing a **Milvus vector database standalone cluster** using Docker Compose. The core functionality is managed by the Python script, **`manage_milvus_infra.py`**.

---

## Quick Start: Management Script

The **`manage_milvus_infra.py`** script provides a robust command-line interface to control the entire Milvus stack, including starting, stopping, and restarting all services with health checks.

### Usage

```
# Start the entire Milvus infrastructure (Milvus, Attu, ETCD, MinIO)
python manage_milvus_infra.py --start

# Stop all running services
python manage_milvus_infra.py --stop

# Restart all services
python manage_milvus_infra.py --restart

```

Upon starting or restarting, the script displays a professional status table for confirmation.

---

## Infrastructure Components & Ports

The environment is built from four essential services:

| Service | Description | Port | Access URL (Example) | 
| ----- | ----- | ----- | ----- | 
| **Milvus Standalone** | Core Vector Database | **19530** | `http://localhost:19530` | 
| **Attu UI** | Web-based interface for Milvus | **8000** | `http://localhost:8000` | 
| **ETCD** | Metadata Storage | 2379 | `http://localhost:2379` | 
| **MinIO** | Object Storage for data files | 9000 | `http://localhost:9000` | 

---

## Data Persistence

All service data is configured to be persisted locally to ensure data integrity across container restarts. Data is stored in the **`volumes/`** directory:

| Component | Data Directory | 
| ----- | ----- | 
| **Milvus** | **`volumes/milvus/`** | 
| **ETCD** | **`volumes/etcd/`** | 
| **MinIO** | **`volumes/minio/`** | 

---

## Prerequisites

To successfully run the management script and deploy the infrastructure, you must have the following installed:

1. **Docker Engine:** Installed and running.

2. **Docker Compose:** Installed.

3. **Python 3.7+**

---

## Error Handling and Recovery

The script is designed for resilience, featuring robust error handling for common issues (e.g., Docker daemon not running, startup failures).

* **Handling Existing Containers:** When starting, the script automatically checks for and attempts to gracefully stop existing containers. If a graceful stop fails, it proceeds to forcefully remove and clean up stubborn containers before initiating a fresh deployment.

* **Graceful Recovery:** The script attempts alternative approaches to complete an operation if an error occurs and provides clear error messages.