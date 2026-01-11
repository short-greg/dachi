import pytest
from pydantic import ValidationError

from dachi.config._models import (
    ActConfig,
    CriterionConfig,
    DachiConfig,
    DifferenceConfig,
    OpsConfig,
    OptimConfig,
    ProcConfig,
)


class TestDifferenceConfig:
    def test_default_values_set(self):
        config = DifferenceConfig()
        assert config.model is None
        assert "Text A" in config.prompt

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            DifferenceConfig(model=123)


class TestDachiConfig:
    def test_nested_structure(self):
        config = DachiConfig(_env_file=None)
        assert isinstance(config.Ops, OpsConfig)
        assert isinstance(config.Proc, ProcConfig)
        assert isinstance(config.Act, ActConfig)
        assert isinstance(config.Inst.Criterion, CriterionConfig)
        assert isinstance(config.Optim, OptimConfig)
