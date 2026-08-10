from unittest.mock import MagicMock, patch

import pytest

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiCancelKernelSessionResponse


class TestKernelsCancel:
    def setup_method(self):
        self.api = KaggleApi.__new__(KaggleApi)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_cancel_sends_the_session_id(self, mock_client):
        response = ApiCancelKernelSessionResponse()
        client = MagicMock()
        client.kernels.kernels_api_client.cancel_kernel_session.return_value = response
        mock_client.return_value.__enter__ = MagicMock(return_value=client)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        result = self.api.kernels_cancel(341475263)

        assert result is response
        request = client.kernels.kernels_api_client.cancel_kernel_session.call_args.args[0]
        assert request.kernel_session_id == 341475263

    @pytest.mark.parametrize("kernel_session_id", [0, -1])
    def test_kernels_cancel_rejects_non_positive_session_ids(self, kernel_session_id):
        with pytest.raises(ValueError, match="must be positive"):
            self.api.kernels_cancel(kernel_session_id)

    @patch.object(KaggleApi, "build_kaggle_client")
    def test_kernels_cancel_surfaces_api_rejection(self, mock_client):
        response = ApiCancelKernelSessionResponse()
        response.error_message = "Session is not active"
        client = MagicMock()
        client.kernels.kernels_api_client.cancel_kernel_session.return_value = response
        mock_client.return_value.__enter__ = MagicMock(return_value=client)
        mock_client.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(ValueError, match="Session is not active"):
            self.api.kernels_cancel(341475263)
