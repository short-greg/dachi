from unittest.mock import MagicMock, patch

import pytest

from dachi.config import reset_config
from dachi.proc._operations import difference


class TestOperationsIntegration:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_difference_uses_config_default(self, monkeypatch):
        monkeypatch.setenv("DACHI_OPS__DIFFERENCE__MODEL", "test-model")
        reset_config()

        with patch("dachi.proc._operations.Engines.get") as mock_get:
            mock_model = MagicMock()
            mock_model.forward.return_value = ("result", None, None)
            mock_get.return_value = mock_model

            difference("a", "b")

            mock_get.assert_called_once_with("test-model")
            called_prompt = mock_model.forward.call_args.kwargs["prompt"]
            assert "Text A" in called_prompt
            assert "a" in called_prompt

    def test_difference_param_override(self):
        with patch("dachi.proc._operations.Engines.get") as mock_get:
            mock_model = MagicMock()
            mock_model.forward.return_value = ("result", None, None)
            mock_get.return_value = mock_model

            difference("a", "b", _model="override-model", _prompt="custom {a} {b}")

            mock_get.assert_called_once_with("override-model")
            called_prompt = mock_model.forward.call_args.kwargs["prompt"]
            assert called_prompt == "custom a b"
