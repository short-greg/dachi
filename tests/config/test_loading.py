import pytest

from dachi.config._models import DachiConfig


class TestYamlLoading:
    def test_load_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "dachi.yaml"
        yaml_file.write_text("Ops:\n  Difference:\n    model: gpt-4\n")

        class FileConfig(DachiConfig):
            model_config = {**DachiConfig.model_config, "yaml_file": str(yaml_file)}

        config = FileConfig(_env_file=None)
        assert config.Ops.Difference.model == "gpt-4"

    def test_missing_yaml_uses_defaults(self):
        config = DachiConfig(_env_file=None)
        assert config.Ops.Difference.model is None
        assert "Text A" in config.Ops.Difference.prompt


class TestEnvVarLoading:
    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("DACHI_OPS__DIFFERENCE__MODEL", "env-model")

        config = DachiConfig(_env_file=None)
        assert config.Ops.Difference.model == "env-model"

    def test_precedence_order(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "dachi.yaml"
        yaml_file.write_text("Ops:\n  Difference:\n    model: yaml-model")
        monkeypatch.setenv("DACHI_OPS__DIFFERENCE__MODEL", "env-model")

        class FileConfig(DachiConfig):
            model_config = {**DachiConfig.model_config, "yaml_file": str(yaml_file)}

        config = FileConfig(_env_file=None)
        assert config.Ops.Difference.model == "env-model"
