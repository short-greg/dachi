from dachi.config import reset_config
from dachi.proc._operations import difference


class DummyModel:
    def __init__(self):
        self.forward_calls = []

    def forward(self, *, prompt, structure=None):
        self.forward_calls.append({"prompt": prompt, "structure": structure})
        return "result", None, None


class TestOperationsIntegration:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_difference_uses_config_default(self, monkeypatch):
        monkeypatch.setenv("DACHI_OPS__DIFFERENCE__MODEL", "test-model")
        reset_config()

        dummy_model = DummyModel()
        monkeypatch.setattr("dachi.proc._operations.Engines.get", lambda key: dummy_model)

        difference("a", "b")

        assert dummy_model.forward_calls
        assert "Text A" in dummy_model.forward_calls[0]["prompt"]
        assert "a" in dummy_model.forward_calls[0]["prompt"]

    def test_difference_param_override(self, monkeypatch):
        dummy_model = DummyModel()
        monkeypatch.setattr("dachi.proc._operations.Engines.get", lambda key: dummy_model)

        difference("a", "b", _model="override-model", _prompt="custom {a} {b}")

        assert dummy_model.forward_calls[0]["prompt"] == "custom a b"
