"""
Comprehensive unit tests for DataValidator.

This module provides systematic testing of the DataValidator class,
including validate_documents and prepare_documents_for_insertion methods.
"""

import pytest

from milvus_ops.collection_operations.schema import CollectionSchema, DataType, FieldSchema
from milvus_ops.data_management_operations.core.validator import DataValidator

# ============================================================================
# Test DataValidator.validate_documents
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestDataValidatorValidateDocuments:
    """
    Test DataValidator.validate_documents method.

    Coverage: Document validation against collection schemas.
    """

    async def test_validate_documents_valid(self, basic_collection_schema):
        """
        Test validate_documents with valid documents.

        Coverage: validate_documents() returns is_valid=True for valid documents.
        """
        # Use documents matching basic_collection_schema (entity_id and vector only)
        documents = [
            {"entity_id": 1, "vector": [0.1] * 128},
            {"entity_id": 2, "vector": [0.2] * 128},
            {"entity_id": 3, "vector": [0.3] * 128},
        ]
        result = await DataValidator.validate_documents(documents, basic_collection_schema)
        assert result.is_valid is True
        assert len(result.errors) == 0

    async def test_validate_documents_valid_dicts(self, basic_collection_schema):
        """
        Test validate_documents with valid dictionary documents.

        Coverage: validate_documents() accepts plain dictionaries.
        """
        # Use documents matching basic_collection_schema (entity_id and vector only)
        documents = [
            {"entity_id": 1, "vector": [0.1] * 128},
            {"entity_id": 2, "vector": [0.2] * 128},
            {"entity_id": 3, "vector": [0.3] * 128},
        ]
        result = await DataValidator.validate_documents(documents, basic_collection_schema)
        assert result.is_valid is True
        assert len(result.errors) == 0

    async def test_validate_documents_missing_required_field(self, basic_collection_schema):
        """
        Test validate_documents with missing required field.

        Coverage: validate_documents() detects missing required fields.
        """
        documents = [{"entity_id": 1}]  # Missing vector field
        result = await DataValidator.validate_documents(documents, basic_collection_schema)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any(
            "Missing required field 'vector'" in str(errors) for errors in result.errors.values()
        )

    async def test_validate_documents_type_mismatch(self, basic_collection_schema):
        """
        Test validate_documents with type mismatch.

        Coverage: validate_documents() detects type mismatches.
        """
        documents = [
            {"entity_id": "invalid_id", "vector": [0.1] * 128}  # String instead of int
        ]
        result = await DataValidator.validate_documents(documents, basic_collection_schema)
        assert result.is_valid is False
        assert len(result.errors) > 0

    async def test_validate_documents_vector_dimension_mismatch(self, basic_collection_schema):
        """
        Test validate_documents with vector dimension mismatch.

        Coverage: validate_documents() detects vector dimension mismatches.
        """
        documents = [{"entity_id": 1, "vector": [0.1] * 64}]  # Wrong dimension (64 instead of 128)
        result = await DataValidator.validate_documents(documents, basic_collection_schema)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("dimension mismatch" in str(errors).lower() for errors in result.errors.values())

    async def test_validate_documents_extraneous_fields_non_dynamic(self, basic_collection_schema):
        """
        Test validate_documents with extraneous fields in non-dynamic schema.

        Coverage: validate_documents() detects extraneous fields when dynamic fields disabled.
        """
        schema = CollectionSchema(
            fields=basic_collection_schema.fields,
            description="Non-dynamic schema",
            enable_dynamic_field=False,
        )
        documents = [{"entity_id": 1, "vector": [0.1] * 128, "extra_field": "value"}]
        result = await DataValidator.validate_documents(documents, schema)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("Unexpected field" in str(errors) for errors in result.errors.values())

    async def test_validate_documents_extraneous_fields_dynamic(self, basic_collection_schema):
        """
        Test validate_documents with extraneous fields in dynamic schema.

        Coverage: validate_documents() allows extraneous fields when dynamic fields enabled.
        """
        schema = CollectionSchema(
            fields=basic_collection_schema.fields,
            description="Dynamic schema",
            enable_dynamic_field=True,
        )
        documents = [{"entity_id": 1, "vector": [0.1] * 128, "extra_field": "value"}]
        result = await DataValidator.validate_documents(documents, schema)
        assert result.is_valid is True
        assert len(result.errors) == 0

    async def test_validate_documents_auto_id_primary_key(self):
        """
        Test validate_documents with auto-id primary key.

        Coverage: validate_documents() allows None ID for auto-id primary keys.
        """
        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name="entity_id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                ),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
            ],
            description="Auto-id schema",
        )
        documents = [{"vector": [0.1] * 128}]  # No ID provided
        result = await DataValidator.validate_documents(documents, schema)
        assert result.is_valid is True
        assert len(result.errors) == 0

    async def test_validate_documents_multiple_errors_per_document(self, basic_collection_schema):
        """
        Test validate_documents with multiple errors per document.

        Coverage: validate_documents() collects all errors for a document.
        """
        documents = [
            {
                "entity_id": "invalid_type",  # Type error
                "vector": [0.1] * 64,  # Dimension error
                "extra_field": "value",  # Extraneous field error (if non-dynamic)
            }
        ]
        schema = CollectionSchema(
            fields=basic_collection_schema.fields,
            description="Non-dynamic schema",
            enable_dynamic_field=False,
        )
        result = await DataValidator.validate_documents(documents, schema)
        assert result.is_valid is False
        assert len(result.errors) > 0
        # Should have multiple errors for the document
        for errors_list in result.errors.values():
            assert len(errors_list) > 1

    async def test_validate_documents_empty_list(self, basic_collection_schema):
        """
        Test validate_documents with empty document list.

        Coverage: validate_documents() handles empty list.
        """
        result = await DataValidator.validate_documents([], basic_collection_schema)
        assert result.is_valid is True
        assert len(result.errors) == 0

    @pytest.mark.parametrize(
        "vector_dim",
        [1, 8, 64, 128, 256, 512, 1024],
    )
    async def test_validate_documents_various_vector_dimensions(self, vector_dim):
        """
        Test validate_documents with various vector dimensions.

        Coverage: validate_documents() validates different vector dimensions.
        """
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="entity_id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            ],
            description="Variable dimension schema",
        )
        documents = [{"entity_id": 1, "vector": [0.1] * vector_dim}]
        result = await DataValidator.validate_documents(documents, schema)
        assert result.is_valid is True

    async def test_validate_documents_with_document_id(self, basic_collection_schema):
        """
        Test validate_documents uses document identifier in errors.

        Coverage: validate_documents() uses document identifier as error key when available.
        """
        # Use dict documents with entity_id to test identifier-based error keys
        # For dict documents without 'id' field, validator uses index as error key
        documents = [{"entity_id": 123, "vector": [0.1] * 128}]  # Valid document
        result = await DataValidator.validate_documents(documents, basic_collection_schema)
        assert result.is_valid is True

        # Invalid document with wrong dimension - validator uses index 0 as error key
        documents = [{"entity_id": 123, "vector": [0.1] * 64}]  # Wrong dimension
        result = await DataValidator.validate_documents(documents, basic_collection_schema)
        assert result.is_valid is False
        # Validator uses document index (0) as error key for dict documents
        assert 0 in result.errors
        assert any("Vector dimension mismatch" in error for error in result.errors[0])


# ============================================================================
# Test DataValidator.prepare_documents_for_insertion
# ============================================================================


@pytest.mark.unit
class TestDataValidatorPrepareDocumentsForInsertion:
    """
    Test DataValidator.prepare_documents_for_insertion method.

    Coverage: Document preparation for Milvus insertion.
    """

    def test_prepare_documents_for_insertion_pydantic_models(
        self, basic_collection_schema, sample_documents
    ):
        """
        Test prepare_documents_for_insertion with Pydantic models.

        Coverage: prepare_documents_for_insertion() converts Pydantic models to dicts.
        """
        result = DataValidator.prepare_documents_for_insertion(
            sample_documents, basic_collection_schema
        )
        assert isinstance(result, list)
        assert len(result) == len(sample_documents)
        assert all(isinstance(doc, dict) for doc in result)
        assert all("entity_id" in doc for doc in result)
        assert all("vector" in doc for doc in result)

    def test_prepare_documents_for_insertion_plain_dicts(
        self, basic_collection_schema, sample_document_dicts
    ):
        """
        Test prepare_documents_for_insertion with plain dictionaries.

        Coverage: prepare_documents_for_insertion() handles plain dictionaries.
        """
        result = DataValidator.prepare_documents_for_insertion(
            sample_document_dicts, basic_collection_schema
        )
        assert isinstance(result, list)
        assert len(result) == len(sample_document_dicts)
        assert all(isinstance(doc, dict) for doc in result)

    def test_prepare_documents_for_insertion_schema_field_ordering(self, basic_collection_schema):
        """
        Test prepare_documents_for_insertion preserves schema field ordering.

        Coverage: prepare_documents_for_insertion() orders fields according to schema.
        """
        documents = [{"entity_id": 1, "vector": [0.1] * 128}]
        result = DataValidator.prepare_documents_for_insertion(documents, basic_collection_schema)
        assert len(result) == 1
        prepared_doc = result[0]
        # Fields should be ordered as in schema
        field_names = list(prepared_doc.keys())
        schema_field_names = [field.name for field in basic_collection_schema.fields]
        assert field_names == schema_field_names

    def test_prepare_documents_for_insertion_missing_fields(self, basic_collection_schema):
        """
        Test prepare_documents_for_insertion with missing fields.

        Coverage: prepare_documents_for_insertion() inserts None for missing fields.
        """
        documents = [{"entity_id": 1}]  # Missing vector field
        result = DataValidator.prepare_documents_for_insertion(documents, basic_collection_schema)
        assert len(result) == 1
        prepared_doc = result[0]
        assert "entity_id" in prepared_doc
        assert "vector" in prepared_doc
        assert prepared_doc["vector"] is None

    def test_prepare_documents_for_insertion_empty_list(self, basic_collection_schema):
        """
        Test prepare_documents_for_insertion with empty list.

        Coverage: prepare_documents_for_insertion() handles empty list.
        """
        result = DataValidator.prepare_documents_for_insertion([], basic_collection_schema)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_prepare_documents_for_insertion_complex_schema(self, complex_collection_schema):
        """
        Test prepare_documents_for_insertion with complex schema.

        Coverage: prepare_documents_for_insertion() handles complex schemas.
        """
        documents = [
            {
                "entity_id": 1,
                "text": "test",
                "embedding": [0.1] * 512,
                "metadata": {"key": "value"},
                "tags": ["tag1", "tag2"],
                "tenant_id": 100,
            }
        ]
        result = DataValidator.prepare_documents_for_insertion(documents, complex_collection_schema)
        assert len(result) == 1
        prepared_doc = result[0]
        # All schema fields should be present
        assert len(prepared_doc) == len(complex_collection_schema.fields)
        assert "entity_id" in prepared_doc
        assert "text" in prepared_doc
        assert "embedding" in prepared_doc
        assert "metadata" in prepared_doc
        assert "tags" in prepared_doc
        assert "tenant_id" in prepared_doc

    def test_prepare_documents_for_insertion_multiple_documents(self, basic_collection_schema):
        """
        Test prepare_documents_for_insertion with multiple documents.

        Coverage: prepare_documents_for_insertion() processes multiple documents.
        """
        documents = [
            {"entity_id": 1, "vector": [0.1] * 128},
            {"entity_id": 2, "vector": [0.2] * 128},
            {"entity_id": 3, "vector": [0.3] * 128},
        ]
        result = DataValidator.prepare_documents_for_insertion(documents, basic_collection_schema)
        assert len(result) == 3
        assert all(isinstance(doc, dict) for doc in result)
        assert result[0]["entity_id"] == 1
        assert result[1]["entity_id"] == 2
        assert result[2]["entity_id"] == 3

    def test_prepare_documents_for_insertion_preserves_values(self, basic_collection_schema):
        """
        Test prepare_documents_for_insertion preserves field values.

        Coverage: prepare_documents_for_insertion() preserves existing field values.
        """
        documents = [{"entity_id": 123, "vector": [0.5, 0.6, 0.7] * 42 + [0.5, 0.6]}]  # 128 dims
        result = DataValidator.prepare_documents_for_insertion(documents, basic_collection_schema)
        assert result[0]["entity_id"] == 123
        assert result[0]["vector"] == documents[0]["vector"]
