"""
Tests for EmbeddingService.

Covered behavior:

- Model initialization
- Document embedding generation
- Query embedding generation
- SentenceTransformer call arguments
- Conversion of model output with tolist()
- Empty and unusual inputs
- Error propagation from the embedding model
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from backend.app.services.embedding_service import (
    EMBEDDING_MODEL_NAME,
    EmbeddingService,
)


class FakeEmbeddingResult:
    """Small array-like test double exposing tolist()."""

    def __init__(self, value: Any) -> None:
        self.value = value
        self.tolist_calls = 0

    def tolist(self) -> Any:
        self.tolist_calls += 1
        return self.value


@pytest.fixture
def mocked_sentence_transformer() -> MagicMock:
    """Return a mocked SentenceTransformer instance."""

    with patch(
        "backend.app.services.embedding_service.SentenceTransformer"
    ) as transformer_class:
        model = MagicMock()
        transformer_class.return_value = model
        yield model


@pytest.fixture
def service(
    mocked_sentence_transformer: MagicMock,
) -> EmbeddingService:
    """Create EmbeddingService without loading a real embedding model."""

    return EmbeddingService()


class TestEmbeddingConfiguration:
    """Tests for embedding-service configuration."""

    def test_embedding_model_name_is_expected_value(self) -> None:
        assert EMBEDDING_MODEL_NAME == "BAAI/bge-small-en-v1.5"

    def test_embedding_model_name_is_non_empty_string(self) -> None:
        assert isinstance(EMBEDDING_MODEL_NAME, str)
        assert EMBEDDING_MODEL_NAME.strip()


class TestInitialization:
    """Tests for EmbeddingService initialization."""

    def test_initializes_sentence_transformer_with_configured_model(
        self,
    ) -> None:
        with patch(
            "backend.app.services.embedding_service.SentenceTransformer"
        ) as transformer_class:
            service = EmbeddingService()

        transformer_class.assert_called_once_with(
            EMBEDDING_MODEL_NAME
        )
        assert service.model is transformer_class.return_value

    def test_each_service_instance_creates_its_own_model(
        self,
    ) -> None:
        first_model = MagicMock()
        second_model = MagicMock()

        with patch(
            "backend.app.services.embedding_service.SentenceTransformer",
            side_effect=[first_model, second_model],
        ) as transformer_class:
            first_service = EmbeddingService()
            second_service = EmbeddingService()

        assert transformer_class.call_count == 2
        assert first_service.model is first_model
        assert second_service.model is second_model

    def test_model_initialization_error_is_propagated(
        self,
    ) -> None:
        with patch(
            "backend.app.services.embedding_service.SentenceTransformer",
            side_effect=RuntimeError("Model unavailable"),
        ):
            with pytest.raises(
                RuntimeError,
                match="Model unavailable",
            ):
                EmbeddingService()


class TestEmbedDocuments:
    """Tests for EmbeddingService.embed_documents()."""

    def test_returns_document_embeddings_as_python_lists(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        expected = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
        encoded = FakeEmbeddingResult(expected)
        mocked_sentence_transformer.encode.return_value = encoded

        result = service.embed_documents(
            [
                "Apple reported revenue growth.",
                "Microsoft disclosed cybersecurity risks.",
            ]
        )

        assert result == expected
        assert encoded.tolist_calls == 1

    def test_calls_encode_with_expected_document_arguments(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        texts = [
            "First document",
            "Second document",
        ]
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult([[0.1], [0.2]])
        )

        service.embed_documents(texts)

        mocked_sentence_transformer.encode.assert_called_once_with(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def test_passes_original_list_object_to_model(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        texts = ["One", "Two"]
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult([[1.0], [2.0]])
        )

        service.embed_documents(texts)

        passed_texts = (
            mocked_sentence_transformer.encode.call_args.args[0]
        )
        assert passed_texts is texts

    def test_empty_document_list_is_forwarded_to_model(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult([])
        )

        result = service.embed_documents([])

        assert result == []
        mocked_sentence_transformer.encode.assert_called_once_with(
            [],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def test_single_document_is_supported(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult([[0.25, 0.75]])
        )

        result = service.embed_documents(
            ["Single document"]
        )

        assert result == [[0.25, 0.75]]

    def test_document_order_is_preserved_from_model_output(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        expected = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult(expected)
        )

        result = service.embed_documents(
            ["First", "Second", "Third"]
        )

        assert result == expected

    def test_high_dimensional_embeddings_are_returned_unchanged(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        vector = [float(index) for index in range(384)]
        expected = [vector]
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult(expected)
        )

        result = service.embed_documents(["Document"])

        assert result == expected
        assert len(result[0]) == 384

    def test_negative_and_zero_values_are_preserved(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        expected = [[-0.5, 0.0, 0.5]]
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult(expected)
        )

        result = service.embed_documents(["Document"])

        assert result == expected

    def test_encode_error_is_propagated(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        mocked_sentence_transformer.encode.side_effect = RuntimeError(
            "Encoding failed"
        )

        with pytest.raises(
            RuntimeError,
            match="Encoding failed",
        ):
            service.embed_documents(["Document"])

    def test_missing_tolist_method_raises_attribute_error(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        mocked_sentence_transformer.encode.return_value = [
            [0.1, 0.2]
        ]

        with pytest.raises(AttributeError):
            service.embed_documents(["Document"])

    def test_tolist_error_is_propagated(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        encoded = MagicMock()
        encoded.tolist.side_effect = ValueError(
            "Conversion failed"
        )
        mocked_sentence_transformer.encode.return_value = encoded

        with pytest.raises(
            ValueError,
            match="Conversion failed",
        ):
            service.embed_documents(["Document"])

    @pytest.mark.parametrize(
        "texts",
        [
            None,
            "single string",
            123,
            {},
            (),
        ],
    )
    def test_unusual_inputs_are_forwarded_without_validation(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
        texts: Any,
    ) -> None:
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult([[0.1]])
        )

        result = service.embed_documents(texts)  # type: ignore[arg-type]

        assert result == [[0.1]]
        mocked_sentence_transformer.encode.assert_called_once_with(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )


class TestEmbedQuery:
    """Tests for EmbeddingService.embed_query()."""

    def test_returns_query_embedding_as_python_list(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        expected = [0.1, 0.2, 0.3]
        encoded = FakeEmbeddingResult(expected)
        mocked_sentence_transformer.encode.return_value = encoded

        result = service.embed_query(
            "What risks did Microsoft disclose?"
        )

        assert result == expected
        assert encoded.tolist_calls == 1

    def test_calls_encode_with_expected_query_arguments(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        query = "What was Apple's revenue?"
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult([0.1, 0.2])
        )

        service.embed_query(query)

        mocked_sentence_transformer.encode.assert_called_once_with(
            query,
            normalize_embeddings=True,
        )

    def test_passes_original_query_to_model(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        query = "Compare Apple and Microsoft."
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult([1.0])
        )

        service.embed_query(query)

        passed_query = (
            mocked_sentence_transformer.encode.call_args.args[0]
        )
        assert passed_query is query

    def test_empty_query_is_forwarded_to_model(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult([0.0, 0.0])
        )

        result = service.embed_query("")

        assert result == [0.0, 0.0]
        mocked_sentence_transformer.encode.assert_called_once_with(
            "",
            normalize_embeddings=True,
        )

    def test_whitespace_query_is_not_trimmed(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        query = "   revenue   "
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult([0.25])
        )

        service.embed_query(query)

        mocked_sentence_transformer.encode.assert_called_once_with(
            query,
            normalize_embeddings=True,
        )

    def test_unicode_query_is_supported(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        query = "Revenue grew by 12.5% — why?"
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult([0.3, 0.7])
        )

        result = service.embed_query(query)

        assert result == [0.3, 0.7]

    def test_high_dimensional_query_embedding_is_returned_unchanged(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        expected = [float(index) for index in range(384)]
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult(expected)
        )

        result = service.embed_query("Revenue")

        assert result == expected
        assert len(result) == 384

    def test_negative_and_zero_values_are_preserved(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        expected = [-0.8, 0.0, 0.8]
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult(expected)
        )

        result = service.embed_query("Risk")

        assert result == expected

    def test_encode_error_is_propagated(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        mocked_sentence_transformer.encode.side_effect = RuntimeError(
            "Query encoding failed"
        )

        with pytest.raises(
            RuntimeError,
            match="Query encoding failed",
        ):
            service.embed_query("Revenue")

    def test_missing_tolist_method_raises_attribute_error(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        mocked_sentence_transformer.encode.return_value = [
            0.1,
            0.2,
        ]

        with pytest.raises(AttributeError):
            service.embed_query("Revenue")

    def test_tolist_error_is_propagated(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        encoded = MagicMock()
        encoded.tolist.side_effect = ValueError(
            "Query conversion failed"
        )
        mocked_sentence_transformer.encode.return_value = encoded

        with pytest.raises(
            ValueError,
            match="Query conversion failed",
        ):
            service.embed_query("Revenue")

    @pytest.mark.parametrize(
        "query",
        [
            None,
            123,
            4.5,
            True,
            [],
            {},
            (),
        ],
    )
    def test_unusual_inputs_are_forwarded_without_validation(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
        query: Any,
    ) -> None:
        mocked_sentence_transformer.encode.return_value = (
            FakeEmbeddingResult([0.1])
        )

        result = service.embed_query(query)  # type: ignore[arg-type]

        assert result == [0.1]
        mocked_sentence_transformer.encode.assert_called_once_with(
            query,
            normalize_embeddings=True,
        )


class TestServiceInteraction:
    """Tests covering both public embedding methods together."""

    def test_document_and_query_calls_use_same_model_instance(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        mocked_sentence_transformer.encode.side_effect = [
            FakeEmbeddingResult([[0.1, 0.2]]),
            FakeEmbeddingResult([0.3, 0.4]),
        ]

        document_result = service.embed_documents(
            ["Financial report"]
        )
        query_result = service.embed_query("Revenue")

        assert document_result == [[0.1, 0.2]]
        assert query_result == [0.3, 0.4]
        assert mocked_sentence_transformer.encode.call_count == 2

    def test_document_and_query_calls_use_different_encode_options(
        self,
        service: EmbeddingService,
        mocked_sentence_transformer: MagicMock,
    ) -> None:
        mocked_sentence_transformer.encode.side_effect = [
            FakeEmbeddingResult([[0.1]]),
            FakeEmbeddingResult([0.2]),
        ]

        service.embed_documents(["Document"])
        service.embed_query("Query")

        document_call = (
            mocked_sentence_transformer.encode.call_args_list[0]
        )
        query_call = (
            mocked_sentence_transformer.encode.call_args_list[1]
        )

        assert document_call.kwargs == {
            "batch_size": 32,
            "show_progress_bar": True,
            "normalize_embeddings": True,
        }
        assert query_call.kwargs == {
            "normalize_embeddings": True,
        }
