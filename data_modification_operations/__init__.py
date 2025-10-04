"""
Data Modification Operations Module

This module handles updating and modifying existing data in Milvus:
- Vector updates by ID
- Metadata field updates
- Bulk update operations
- Delete operations (by ID and by expression)
- Data validation during updates
- Transaction management for data consistency
- Optimistic and pessimistic locking strategies

Provides reliable data modification capabilities with proper
error handling and consistency guarantees for production use.
"""
