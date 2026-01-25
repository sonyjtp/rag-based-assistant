"""
Unit tests for embedding model initialization.
Tests device detection, configuration, and model setup.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.embeddings import initialize_embedding_model


@pytest.fixture
def embedding_mocks():
    """Fixture providing mocked embedding components."""
    with patch("src.embeddings.HuggingFaceEmbeddings") as mock_embeddings, patch(
        "src.embeddings.torch.cuda.is_available"
    ) as mock_cuda, patch(
        "src.embeddings.torch.backends.mps.is_available"
    ) as mock_mps, patch(
        "src.embeddings.logger"
    ) as mock_logger:
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance
        yield {
            "embeddings": mock_embeddings,
            "cuda": mock_cuda,
            "mps": mock_mps,
            "instance": mock_instance,
            "logger": mock_logger,
        }


# pylint: disable=redefined-outer-name
class TestEmbeddingModelInitialization:
    """Test embedding model initialization with device detection."""

    @pytest.mark.parametrize(
        "cuda_available,mps_available,expected_device",
        [
            (True, True, "cuda"),  # CUDA priority over MPS
            (True, False, "cuda"),  # CUDA available
            (False, True, "mps"),  # MPS fallback
            (False, False, "cpu"),  # CPU fallback
        ],
    )
    def test_device_detection_priority(
        self, embedding_mocks, cuda_available, mps_available, expected_device
    ):
        """Parametrized test for device selection priority: CUDA > MPS > CPU."""
        embedding_mocks["cuda"].return_value = cuda_available
        embedding_mocks["mps"].return_value = mps_available

        result = initialize_embedding_model()

        # Verify device was selected with correct priority
        embedding_mocks["embeddings"].assert_called_once()
        call_kwargs = embedding_mocks["embeddings"].call_args[1]
        assert call_kwargs["model_kwargs"]["device"] == expected_device
        assert result == embedding_mocks["instance"]

    def test_default_model_name(self, embedding_mocks):
        """Test that default model name is used when not configured."""
        embedding_mocks["cuda"].return_value = False
        embedding_mocks["mps"].return_value = False

        initialize_embedding_model()

        call_kwargs = embedding_mocks["embeddings"].call_args[1]
        assert call_kwargs["model_name"] == "sentence-transformers/all-mpnet-base-v2"

    @patch("src.embeddings.VECTOR_DB_EMBEDDING_MODEL", "custom-model-name")
    def test_custom_model_from_config(self, embedding_mocks):
        """Test that custom model name from config is used."""
        embedding_mocks["cuda"].return_value = False
        embedding_mocks["mps"].return_value = False

        initialize_embedding_model()

        call_kwargs = embedding_mocks["embeddings"].call_args[1]
        assert call_kwargs["model_name"] == "custom-model-name"

    def test_returns_embedding_instance(self, embedding_mocks):
        """Test that function returns HuggingFaceEmbeddings instance."""
        embedding_mocks["cuda"].return_value = False
        embedding_mocks["mps"].return_value = False

        result = initialize_embedding_model()

        assert result == embedding_mocks["instance"]
        assert result is not None

    def test_model_kwargs_structure(self, embedding_mocks):
        """Test that model_kwargs includes device configuration."""
        embedding_mocks["cuda"].return_value = False
        embedding_mocks["mps"].return_value = True

        initialize_embedding_model()

        call_kwargs = embedding_mocks["embeddings"].call_args[1]
        assert "model_kwargs" in call_kwargs
        assert isinstance(call_kwargs["model_kwargs"], dict)
        assert "device" in call_kwargs["model_kwargs"]
        assert call_kwargs["model_kwargs"]["device"] == "mps"

    def test_device_logging(self, embedding_mocks):
        """Test that device selection is logged."""
        embedding_mocks["cuda"].return_value = True
        embedding_mocks["mps"].return_value = False

        initialize_embedding_model()

        # Verify logging was called
        embedding_mocks["logger"].info.assert_called()
        log_message = embedding_mocks["logger"].info.call_args[0][0]
        assert "device" in log_message.lower()
        assert "cuda" in log_message.lower()

    def test_multiple_calls_create_new_instances(self, embedding_mocks):
        """Test that multiple calls create independent instances."""
        embedding_mocks["cuda"].return_value = False
        embedding_mocks["mps"].return_value = False
        instance_1, instance_2 = MagicMock(), MagicMock()
        embedding_mocks["embeddings"].side_effect = [instance_1, instance_2]

        result_1 = initialize_embedding_model()
        result_2 = initialize_embedding_model()

        assert result_1 == instance_1
        assert result_2 == instance_2
        assert result_1 is not result_2
        assert embedding_mocks["embeddings"].call_count == 2

    def test_cuda_check_called_first(self, embedding_mocks):
        """Test that CUDA availability is checked first."""
        embedding_mocks["cuda"].return_value = True
        embedding_mocks["mps"].return_value = False

        initialize_embedding_model()

        # CUDA should be checked, and since it's available, MPS should not be checked
        embedding_mocks["cuda"].assert_called()
        assert (
            embedding_mocks["mps"].call_count == 0
        )  # MPS not checked when CUDA available

    def test_mps_checked_when_cuda_unavailable(self, embedding_mocks):
        """Test that MPS availability is checked when CUDA is not available."""
        embedding_mocks["cuda"].return_value = False
        embedding_mocks["mps"].return_value = True

        initialize_embedding_model()

        # Both should be checked since CUDA is False
        embedding_mocks["cuda"].assert_called()
        embedding_mocks["mps"].assert_called()

    @pytest.mark.parametrize(
        "cuda,mps",
        [
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ],
    )
    def test_all_device_combinations(self, embedding_mocks, cuda, mps):
        """Test all combinations of CUDA and MPS availability."""
        embedding_mocks["cuda"].return_value = cuda
        embedding_mocks["mps"].return_value = mps

        result = initialize_embedding_model()

        assert result is not None
        embedding_mocks["embeddings"].assert_called_once()
