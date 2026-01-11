from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource
class DifferenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str | None = None
    prompt: str = (
        "Describe everything that is in Text A but is not in Text B.\n"
        "Text A:\n"
        "{a}\n"
        "Text B:\n"
        "{b}"
    )


class SymmetricDifferenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str | None = None
    prompt: str = (
        "Describe everything that is in Text A but is not in Text B, and everything that is in Text B but is not in Text A.\n"
        "Text A:\n"
        "{a}\n"
        "Text B:\n"
        "{b}"
    )


class UnionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str | None = None
    prompt: str = "Combine the following texts into one unified text, separating sections with '{sep}': \n{texts}"
    sep: str = "\n"
    header: str = "**TEXT {}**\n"


class IntersectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str | None = None
    prompt: str = "Given the following texts, extract and return only the content that is common to all texts.\nTexts:\n{texts}"
    sep: str = "\n"
    header: str = "**TEXT {}**\n"


class OpsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    Difference: DifferenceConfig = Field(default_factory=DifferenceConfig)
    SymmetricDifference: SymmetricDifferenceConfig = Field(default_factory=SymmetricDifferenceConfig)
    Union: UnionConfig = Field(default_factory=UnionConfig)
    Intersection: IntersectionConfig = Field(default_factory=IntersectionConfig)


class AsyncDispatcherConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_concurrency: int = 8


class ToolUserConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_iterations: int = 10


class ParserConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    csv_delimiter: str = ","
    char_delimiter: str = ","
    line_separator: str = "\n"


class ProcConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    AsyncDispatcher: AsyncDispatcherConfig = Field(default_factory=AsyncDispatcherConfig)
    ToolUser: ToolUserConfig = Field(default_factory=ToolUserConfig)
    Parser: ParserConfig = Field(default_factory=ParserConfig)


class StateChartConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    queue_maxsize: int = 1024
    queue_overflow: Literal["drop_newest", "drop_oldest", "block"] = "drop_newest"
    checkpoint_policy: Literal["yield", "hard"] = "yield"
    emit_enforcement: Literal["none", "warn", "error"] = "warn"
    auto_finish: bool = True


class ParallelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    fails_on: int = 1
    succeeds_on: int = -1
    success_priority: bool = True
    preempt: bool = False
    auto_run: bool = True


class ActConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    StateChart: StateChartConfig = Field(default_factory=StateChartConfig)
    Parallel: ParallelConfig = Field(default_factory=ParallelConfig)


class CriterionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    likert_min: int = 1
    likert_max: int = 5
    score_min: float = 0.0
    score_max: float = 10.0
    membership_min: float = 0.0
    membership_max: float = 1.0


class InstConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    Criterion: CriterionConfig = Field(default_factory=CriterionConfig)


class LangOptimConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_template: str = "<default>"


class LangCriticConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_template: str = "<default>"


class OptimConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    LangOptim: LangOptimConfig = Field(default_factory=LangOptimConfig)
    LangCritic: LangCriticConfig = Field(default_factory=LangCriticConfig)


class EnginesConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_model: str | None = None


class DachiConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DACHI_",
        env_nested_delimiter="__",
        yaml_file=[
            "dachi.yaml",
            "dachi.yml",
            ".dachi.yaml",
            ".dachi.yml",
            "~/.dachi/config.yaml",
            "~/.config/dachi/config.yaml",
        ],
        frozen=True,
    )

    Ops: OpsConfig = Field(default_factory=OpsConfig)
    Proc: ProcConfig = Field(default_factory=ProcConfig)
    Act: ActConfig = Field(default_factory=ActConfig)
    Inst: InstConfig = Field(default_factory=InstConfig)
    Optim: OptimConfig = Field(default_factory=OptimConfig)
    Engines: EnginesConfig = Field(default_factory=EnginesConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        yaml_file = getattr(settings_cls, "_yaml_file_override", None) or settings_cls.model_config.get("yaml_file")
        yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file)
        return (
            init_settings,
            env_settings,
            yaml_settings,
            dotenv_settings,
            file_secret_settings,
        )
