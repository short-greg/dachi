import pytest

from dachi.config import ConfigAccessor
from dachi.config._models import DachiConfig


class TestConfigAccessor:
    def test_attribute_access(self):
        config = ConfigAccessor(DachiConfig(_env_file=None))
        assert config.Ops.Difference.model is None

    def test_dict_access(self):
        config = ConfigAccessor(DachiConfig(_env_file=None))
        assert config["Ops"].Difference.model is None

    def test_full_dict_access(self):
        config = ConfigAccessor(DachiConfig(_env_file=None))
        assert "Text A" in config["Ops"]["Difference"]["prompt"]

    def test_freeze_prevents_modification(self):
        config = ConfigAccessor(DachiConfig(_env_file=None))
        config.freeze()
        with pytest.raises(RuntimeError):
            config.Ops.Difference.model = "new"
