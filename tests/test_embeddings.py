"""
Unit tests for embedding model initialization.
Tests device detection, configuration, and model setup.
"""

import pytest
from unittest.mock import patch, MagicMock
import torch
from src.embeddings import initialize_embedding_model


class TestEmbeddingModelInitialization:
    """Test embedding model initialization with device detection."""

    @patch('src.embeddings.HuggingFaceEmbeddings')
    @patch('src.embeddings.torch.cuda.is_available')
    @patch('src.embeddings.torch.backends.mps.is_available')
    def test_initialize_with_cuda(self, mock_mps_available, mock_cuda_available, mock_embeddings):
        """Test initialization when CUDA is available."""
        # Setup: CUDA available, MPS not needed
        mock_cuda_available.return_value = True
        mock_mps_available.return_value = False
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance

        result = initialize_embedding_model()

        # Verify CUDA was selected
        mock_embeddings.assert_called_once()
        call_kwargs = mock_embeddings.call_args[1]
        assert call_kwargs['model_kwargs']['device'] == 'cuda'
        assert result == mock_embedding_instance

    @patch('src.embeddings.HuggingFaceEmbeddings')
    @patch('src.embeddings.torch.cuda.is_available')
    @patch('src.embeddings.torch.backends.mps.is_available')
    def test_initialize_with_mps(self, mock_mps_available, mock_cuda_available, mock_embeddings):
        """Test initialization when MPS (Apple Silicon) is available."""
        # Setup: CUDA not available, MPS available
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance

        result = initialize_embedding_model()

        # Verify MPS was selected
        mock_embeddings.assert_called_once()
        call_kwargs = mock_embeddings.call_args[1]
        assert call_kwargs['model_kwargs']['device'] == 'mps'
        assert result == mock_embedding_instance

    @patch('src.embeddings.HuggingFaceEmbeddings')
    @patch('src.embeddings.torch.cuda.is_available')
    @patch('src.embeddings.torch.backends.mps.is_available')
    def test_initialize_with_cpu(self, mock_mps_available, mock_cuda_available, mock_embeddings):
        """Test initialization when only CPU is available (fallback)."""
        # Setup: Neither CUDA nor MPS available
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance

        result = initialize_embedding_model()

        # Verify CPU was selected
        mock_embeddings.assert_called_once()
        call_kwargs = mock_embeddings.call_args[1]
        assert call_kwargs['model_kwargs']['device'] == 'cpu'
        assert result == mock_embedding_instance

    @patch.dict('os.environ', {'VECTOR_DB_EMBEDDING_MODEL': 'custom-model-name'})
    @patch('src.embeddings.HuggingFaceEmbeddings')
    @patch('src.embeddings.torch.cuda.is_available')
    @patch('src.embeddings.torch.backends.mps.is_available')
    def test_initialize_with_custom_model(self, mock_mps_available, mock_cuda_available, mock_embeddings):
        """Test initialization with custom model from environment variable."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance

        result = initialize_embedding_model()

        # Verify custom model name was used
        mock_embeddings.assert_called_once()
        call_kwargs = mock_embeddings.call_args[1]
        assert call_kwargs['model_name'] == 'custom-model-name'

    @patch('src.embeddings.HuggingFaceEmbeddings')
    @patch('src.embeddings.torch.cuda.is_available')
    @patch('src.embeddings.torch.backends.mps.is_available')
    @patch.dict('os.environ', {}, clear=True)
    def test_initialize_with_default_model(self, mock_mps_available, mock_cuda_available, mock_embeddings):
        """Test initialization uses default model when no env var is set."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance

        result = initialize_embedding_model()

        # Verify default model name was used
        mock_embeddings.assert_called_once()
        call_kwargs = mock_embeddings.call_args[1]
        assert call_kwargs['model_name'] == 'sentence-transformers/all-mpnet-base-v2'

    @patch('src.embeddings.HuggingFaceEmbeddings')
    @patch('src.embeddings.torch.cuda.is_available')
    @patch('src.embeddings.torch.backends.mps.is_available')
    def test_initialize_returns_embeddings_instance(self, mock_mps_available, mock_cuda_available, mock_embeddings):
        """Test that function returns HuggingFaceEmbeddings instance."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        mock_embedding_instance = MagicMock()
        mock_embedding_instance.__class__ = MagicMock
        mock_embeddings.return_value = mock_embedding_instance

        result = initialize_embedding_model()

        # Verify return type
        assert result is not None
        assert result == mock_embedding_instance

    @patch('src.embeddings.HuggingFaceEmbeddings')
    @patch('src.embeddings.torch.cuda.is_available')
    @patch('src.embeddings.torch.backends.mps.is_available')
    def test_device_priority_cuda_over_mps(self, mock_mps_available, mock_cuda_available, mock_embeddings):
        """Test that CUDA has priority over MPS if both available."""
        # Setup: Both CUDA and MPS available (CUDA should win)
        mock_cuda_available.return_value = True
        mock_mps_available.return_value = True
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance

        result = initialize_embedding_model()

        # Verify CUDA was selected (not MPS)
        call_kwargs = mock_embeddings.call_args[1]
        assert call_kwargs['model_kwargs']['device'] == 'cuda'

    @patch('src.embeddings.HuggingFaceEmbeddings')
    @patch('src.embeddings.torch.cuda.is_available')
    @patch('src.embeddings.torch.backends.mps.is_available')
    def test_model_kwargs_structure(self, mock_mps_available, mock_cuda_available, mock_embeddings):
        """Test that model_kwargs are properly structured."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance

        result = initialize_embedding_model()

        # Verify model_kwargs structure
        call_kwargs = mock_embeddings.call_args[1]
        assert 'model_kwargs' in call_kwargs
        assert isinstance(call_kwargs['model_kwargs'], dict)
        assert 'device' in call_kwargs['model_kwargs']

    @patch('src.embeddings.HuggingFaceEmbeddings')
    @patch('src.embeddings.torch.cuda.is_available')
    @patch('src.embeddings.torch.backends.mps.is_available')
    def test_device_passed_to_embeddings(self, mock_mps_available, mock_cuda_available, mock_embeddings):
        """Test that detected device is passed to HuggingFaceEmbeddings."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance

        result = initialize_embedding_model()

        # Verify device parameter was passed
        call_kwargs = mock_embeddings.call_args[1]
        device = call_kwargs['model_kwargs']['device']
        assert device in ['cuda', 'mps', 'cpu']

    @patch('src.embeddings.HuggingFaceEmbeddings')
    @patch('src.embeddings.torch.cuda.is_available')
    @patch('src.embeddings.torch.backends.mps.is_available')
    @patch('src.embeddings.logger')
    def test_logging_device_info(self, mock_logger, mock_mps_available, mock_cuda_available, mock_embeddings):
        """Test that device information is logged."""
        mock_cuda_available.return_value = True
        mock_mps_available.return_value = False
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance

        result = initialize_embedding_model()

        # Verify logging was called
        assert mock_logger.info.called
        call_args = mock_logger.info.call_args[0][0]
        assert 'device' in call_args.lower()
        assert 'cuda' in call_args.lower()

    @patch('src.embeddings.HuggingFaceEmbeddings')
    @patch('src.embeddings.torch.cuda.is_available')
    @patch('src.embeddings.torch.backends.mps.is_available')
    def test_multiple_calls_return_different_instances(self, mock_mps_available, mock_cuda_available, mock_embeddings):
        """Test that multiple calls create separate instances."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        mock_instance_1 = MagicMock()
        mock_instance_2 = MagicMock()
        mock_embeddings.side_effect = [mock_instance_1, mock_instance_2]

        result_1 = initialize_embedding_model()
        result_2 = initialize_embedding_model()

        # Verify two separate instances were created
        assert result_1 == mock_instance_1
        assert result_2 == mock_instance_2
        assert result_1 is not result_2
        assert mock_embeddings.call_count == 2


class TestEmbeddingModelDeviceDetection:
    """Test device detection logic in detail."""

    @patch('src.embeddings.HuggingFaceEmbeddings')
    @patch('src.embeddings.torch.cuda.is_available')
    @patch('src.embeddings.torch.backends.mps.is_available')
    def test_cuda_priority_order(self, mock_mps_available, mock_cuda_available, mock_embeddings):
        """Test that device selection follows correct priority: CUDA > MPS > CPU."""
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance

        # Test all combinations
        test_cases = [
            (True, True, 'cuda'),    # CUDA available → use CUDA
            (True, False, 'cuda'),   # CUDA available → use CUDA
            (False, True, 'mps'),    # Only MPS available → use MPS
            (False, False, 'cpu'),   # Neither available → use CPU
        ]

        for cuda_avail, mps_avail, expected_device in test_cases:
            mock_cuda_available.return_value = cuda_avail
            mock_mps_available.return_value = mps_avail
            mock_embeddings.reset_mock()

            initialize_embedding_model()

            call_kwargs = mock_embeddings.call_args[1]
            actual_device = call_kwargs['model_kwargs']['device']
            assert actual_device == expected_device, \
                f"CUDA={cuda_avail}, MPS={mps_avail}: expected {expected_device}, got {actual_device}"

    @patch('src.embeddings.HuggingFaceEmbeddings')
    @patch('src.embeddings.torch.cuda.is_available')
    @patch('src.embeddings.torch.backends.mps.is_available')
    def test_device_parameter_format(self, mock_mps_available, mock_cuda_available, mock_embeddings):
        """Test that device parameter is in correct format."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance

        initialize_embedding_model()

        call_kwargs = mock_embeddings.call_args[1]
        device = call_kwargs['model_kwargs']['device']

        # Device should be a string, not an object
        assert isinstance(device, str)
        # Device should be lowercase
        assert device == device.lower()
