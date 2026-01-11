# Spec: Dachi Configuration System

**Date:** 2026-01-11
**Status:** Draft
**Author:** Greg Short (with Claude assistance)

---

## 1. Executive Summary

### Goal
Create a flexible, type-safe configuration system for Dachi that allows users to configure defaults for operations (prompts, models, parameters) via YAML files and environment variables, while maintaining backward compatibility with runtime function parameter overrides.

### Key Features
- **Dual access patterns**: `dachi.config.Ops.Difference.model` AND `dachi.config['Ops'].Difference.model`
- **YAML file loading** with automatic discovery
- **Environment variable overrides** with precedence (env vars > YAML > defaults)
- **Pydantic validation** for type safety and clear error messages
- **Initialization-only** (immutable after load) - users override via function parameters
- **Sensible defaults** from current hardcoded values in codebase

### Approach
Use **`pydantic-settings`** library (official Pydantic extension) instead of building from scratch. This provides:
- Built-in YAML support via `YamlConfigSettingsSource`
- Automatic environment variable parsing with nested delimiter support (`__`)
- Validation and type coercion
- Well-tested, maintained solution

---

## 2. User Stories

### Story 1: Developer Sets Default Model via YAML
**As a** Dachi user
**I want to** specify default LLM models in a YAML file
**So that** I don't have to pass `_model` to every function call

**Acceptance Criteria:**
- ✅ Can create `dachi.yaml` with `Ops.Difference.model: gpt-4`
- ✅ Calling `difference(a, b)` without `_model` parameter uses `gpt-4`
- ✅ Can still override with `difference(a, b, _model='claude-3')`

**Example:**
```yaml
# dachi.yaml
Ops:
  Difference:
    model: gpt-4
```

```python
from dachi.proc import difference
result = difference("text a", "text b")  # Uses gpt-4 from config
result = difference("text a", "text b", _model="claude-3")  # Override
```

---

### Story 2: DevOps Engineer Overrides Config with Environment Variables
**As a** DevOps engineer
**I want to** override configuration via environment variables
**So that** I can deploy the same code with different configs per environment

**Acceptance Criteria:**
- ✅ Setting `DACHI_OPS__DIFFERENCE__MODEL=gpt-4-turbo` overrides YAML and defaults
- ✅ Environment variables take highest precedence
- ✅ Can override any config value, not just models

**Example:**
```bash
export DACHI_OPS__DIFFERENCE__MODEL=gpt-4-turbo
export DACHI_ENGINES__DEFAULT_MODEL=claude-3
python my_script.py  # Uses env var values
```

---

### Story 3: Data Scientist Customizes Prompts
**As a** data scientist
**I want to** customize default prompts for operations
**So that** I can optimize prompt engineering once and reuse everywhere

**Acceptance Criteria:**
- ✅ Can set `Ops.Difference.prompt` in YAML with custom template
- ✅ Template supports placeholders like `{a}` and `{b}`
- ✅ Operations use custom prompt by default
- ✅ Can override at runtime with `_prompt` parameter

**Example:**
```yaml
Ops:
  Difference:
    prompt: |
      Compare these texts and identify unique elements in Text A.

      Text A:
      {a}

      Text B:
      {b}
```

---

### Story 4: Developer Accesses Config Flexibly
**As a** developer
**I want to** access config using both attribute and dict syntax
**So that** I can use the style that fits my code context

**Acceptance Criteria:**
- ✅ `config.Ops.Difference.model` works (attribute access)
- ✅ `config['Ops'].Difference.model` works (dict access)
- ✅ `config['Ops']['Difference']['model']` works (full dict access)
- ✅ All three return the same value

**Example:**
```python
from dachi.config import config

# All equivalent
model1 = config.Ops.Difference.model
model2 = config['Ops'].Difference.model
model3 = config['Ops']['Difference']['model']
```

---

### Story 5: Developer Gets Type Safety and IDE Support
**As a** developer
**I want to** get autocomplete and type checking for config
**So that** I catch errors early and write code faster

**Acceptance Criteria:**
- ✅ Pydantic validates config types on load
- ✅ Invalid config raises clear validation errors with field names
- ✅ IDE autocomplete works for config attributes (where supported)

**Example:**
```yaml
# Invalid config
Ops:
  Difference:
    model: 123  # Should be string or null
```

```python
# Raises ValidationError:
# validation error for DifferenceConfig
#   model
#     Input should be a valid string [type=string_type, ...]
```

---

### Story 6: New Module Author Extends Config
**As a** framework contributor
**I want to** easily add new config sections for new modules
**So that** the config system scales with the framework

**Acceptance Criteria:**
- ✅ Can add new namespace (e.g., `Act` for behavior trees) by adding Pydantic model
- ✅ New config automatically supports YAML, env vars, and dual access
- ✅ Clear pattern to follow from existing namespaces

**Example:**
```python
# In dachi/config/_models.py
class ActConfig(BaseModel):
    BehaviorTree: BehaviorTreeConfig = Field(default_factory=BehaviorTreeConfig)

class DachiConfig(BaseSettings):
    Ops: OpsConfig = Field(default_factory=OpsConfig)
    Act: ActConfig = Field(default_factory=ActConfig)  # Added
    # ... rest
```

---

### Story 7: Application Developer Adds Custom Config
**As an** application developer using Dachi
**I want to** add application settings alongside Dachi config
**So that** I can manage both without depending on Dachi internals

**Acceptance Criteria:**
- ✅ Recommended path: keep app config separate (composition) with its own env prefix and YAML file
- ✅ Dachi config remains unchanged and continues to work as-is
- ✅ Optional advanced path: can extend `DachiConfig` for a single config object when needed
- ✅ Both patterns support environment variables and YAML loading

**Example:**
**Recommended (Composition):** keep Dachi and app config separate.

```python
# my_app/config.py
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class DatabaseConfig(BaseModel):
    url: str = "sqlite:///app.db"
    pool_size: int = 5

class AppSettings(BaseSettings):
    """Application settings (separate from Dachi)."""
    model_config = SettingsConfigDict(
        env_prefix="MYAPP_",
        yaml_file="myapp.yaml",
    )

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    debug: bool = False
```

```python
# Usage
from dachi.config import config as dachi_config
from my_app.config import AppSettings

app_config = AppSettings()

# Two separate config objects
dachi_config.Ops.Difference.model  # Dachi settings
app_config.database.url            # App settings
```

```yaml
# dachi.yaml
Ops:
  Difference:
    model: gpt-4

# myapp.yaml
database:
  url: "postgresql://localhost/prod"
  pool_size: 5
debug: false
```

**Optional (Single config via inheritance):** advanced users can still extend `DachiConfig` to merge settings.

```python
# In my_app/config.py
from dachi.config import DachiConfig as BaseDachiConfig
from pydantic import BaseModel, Field

class MyAppConfig(BaseModel):
    database_url: str = "sqlite:///app.db"
    max_retries: int = 3
    timeout: float = 30.0

class AppConfig(BaseDachiConfig):
    """Extended config with my application settings."""
    MyApp: MyAppConfig = Field(default_factory=MyAppConfig)

# In my_app/main.py
from dachi.config import init_config
from my_app.config import AppConfig

config = init_config(config_class=AppConfig)
print(config.MyApp.database_url)  # Access custom config
print(config.Ops.Difference.model)  # Access Dachi config
```

---

## 3. Architecture

### 3.1 Technology Choice: Pydantic Settings

**Decision:** Use [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) library instead of building custom solution.

**Rationale:**
- ✅ **Official Pydantic extension** - well-maintained, documented
- ✅ **Built-in YAML support** via `YamlConfigSettingsSource`
- ✅ **Env var parsing** with nested delimiter (`__`) for nested configs
- ✅ **Validation** - automatic type coercion and clear error messages
- ✅ **Battle-tested** - used by FastAPI, many production projects
- ✅ **Reduces implementation** - ~70% less code than custom solution
- ✅ **Future-proof** - supports TOML, JSON, secrets files easily

**Sources:**
- [Settings Management - Pydantic Validation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [pydantic-settings-yaml · PyPI](https://pypi.org/project/pydantic-settings-yaml/)
- [How to Load Configuration in Pydantic](https://medium.com/@wihlarkop/how-to-load-configuration-in-pydantic-3693d0ee81a3)

---

### 3.2 Config Schema Hierarchy

```
DachiConfig (BaseSettings root)
├── Ops (OpsConfig) - Text operations
│   ├── Difference (DifferenceConfig)
│   │   ├── model: str | None = None
│   │   └── prompt: str = "<default template>"
│   ├── SymmetricDifference (SymmetricDifferenceConfig)
│   │   ├── model: str | None = None
│   │   └── prompt: str = "<default template>"
│   ├── Union (UnionConfig)
│   │   ├── model: str | None = None
│   │   ├── prompt: str = "<default template>"
│   │   ├── sep: str = '\n'
│   │   └── header: str = "**TEXT {}**\n"
│   └── Intersection (IntersectionConfig)
│       ├── model: str | None = None
│       ├── prompt: str = "<default template>"
│       ├── sep: str = '\n'
│       └── header: str = "**TEXT {}**\n"
├── Proc (ProcConfig) - Processing settings
│   ├── AsyncDispatcher (AsyncDispatcherConfig)
│   │   └── max_concurrency: int = 8
│   ├── ToolUser (ToolUserConfig)
│   │   └── max_iterations: int = 10
│   └── Parser (ParserConfig)
│       ├── csv_delimiter: str = ','
│       ├── char_delimiter: str = ','
│       └── line_separator: str = '\n'
├── Act (ActConfig) - Behavior tree and state chart settings
│   ├── StateChart (StateChartConfig)
│   │   ├── queue_maxsize: int = 1024
│   │   ├── queue_overflow: Literal["drop_newest", "drop_oldest", "block"] = "drop_newest"
│   │   ├── checkpoint_policy: Literal["yield", "hard"] = "yield"
│   │   ├── emit_enforcement: Literal["none", "warn", "error"] = "warn"
│   │   └── auto_finish: bool = True
│   └── Parallel (ParallelConfig)
│       ├── fails_on: int = 1
│       ├── succeeds_on: int = -1
│       ├── success_priority: bool = True
│       ├── preempt: bool = False
│       └── auto_run: bool = True
├── Inst (InstConfig) - Instruction and criterion settings
│   └── Criterion (CriterionConfig)
│       ├── likert_min: int = 1
│       ├── likert_max: int = 5
│       ├── score_min: float = 0.0
│       ├── score_max: float = 10.0
│       ├── membership_min: float = 0.0
│       └── membership_max: float = 1.0
├── Optim (OptimConfig) - Optimization settings
│   ├── LangOptim (LangOptimConfig)
│   │   └── prompt_template: str = "<default>"
│   └── LangCritic (LangCriticConfig)
│       └── prompt_template: str = "<default>"
└── Engines (EnginesConfig) - Global engine settings
    └── default_model: str | None = None
```

**User Extensibility:** Applications can subclass `DachiConfig` to add custom namespaces (see Story 7).

---

### 3.3 Loading Precedence

**Priority (highest to lowest):**
1. **Environment variables** (`DACHI_*`)
2. **YAML file** (if found)
3. **Pydantic Field defaults** (hardcoded in models)

**YAML Search Paths (in order):**
1. `./dachi.yaml`
2. `./dachi.yml`
3. `./.dachi.yaml`
4. `./.dachi.yml`
5. `~/.dachi/config.yaml`
6. `~/.config/dachi/config.yaml`

Implemented via `YamlConfigSettingsSource` in `settings_customise_sources()`.

---

### 3.4 Environment Variable Naming Convention

**Format:** `DACHI_<NAMESPACE>__<CLASS>__<FIELD>`

**Note:** Double underscore `__` is the nested delimiter (configurable via `env_nested_delimiter='__'`)

**Examples:**
- `DACHI_OPS__DIFFERENCE__MODEL=gpt-4`
- `DACHI_OPS__UNION__SEP=$'\n\n'`
- `DACHI_ENGINES__DEFAULT_MODEL=claude-3`
- `DACHI_OPTIM__LANGOPTIM__PROMPT_TEMPLATE="..."`

**Parsing Rules (automatic via pydantic-settings):**
- Case-insensitive environment variable names
- Nested fields use `__` delimiter
- Automatic type coercion (str → int, str → bool, etc.)
- `none`, `null`, `""` → None

---

### 3.5 Application Extension Patterns

**Decision:** Recommend composition for application settings; keep inheritance as an advanced option; defer plugin/registry pattern.

**Options:**
- **Composition (recommended):** Keep Dachi config and app config separate. Each gets its own env prefix and YAML file. No dependency on Dachi internals, clearer separation of concerns.
- **Single config via inheritance (optional):** Extend `DachiConfig` to add app fields when a single object is required. Supports YAML and env vars but couples app to Dachi config schema.
- **Plugin/registry (future):** Could register extra config namespaces at runtime, but not needed for initial release.

**Implications:**
- Documentation will show composition first and mark inheritance as advanced.
- No breaking changes required in Dachi to support composition.
- Extension registry remains out of scope for this spec.

---

### 3.6 Component Design

#### 1. Config Models (`dachi/config/_models.py`)
- Pydantic `BaseModel` subclasses for nested configs
- Root `DachiConfig` extends `BaseSettings` (from pydantic-settings)
- Field defaults match current hardcoded values from `_operations.py`

```python
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class DifferenceConfig(BaseModel):
    model: str | None = None
    prompt: str = "Describe everything that is in Text A but is not in Text B.\nText A:\n{a}\nText B:\n{b}"

class OpsConfig(BaseModel):
    Difference: DifferenceConfig = Field(default_factory=DifferenceConfig)
    # ... other operations

class DachiConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='DACHI_',
        env_nested_delimiter='__',
        yaml_file=['dachi.yaml', 'dachi.yml', '.dachi.yaml', ...],
    )

    Ops: OpsConfig = Field(default_factory=OpsConfig)
    Optim: OptimConfig = Field(default_factory=OptimConfig)
    Engines: EnginesConfig = Field(default_factory=EnginesConfig)
```

#### 2. YAML Loading (`settings_customise_sources`)
Uses `YamlConfigSettingsSource` from pydantic-settings:

```python
from pydantic_settings import YamlConfigSettingsSource

class DachiConfig(BaseSettings):
    @classmethod
    def settings_customise_sources(cls, ...):
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls=cls),
            default_settings,
        )
```

#### 3. Dual Access Wrapper (`dachi/config/_accessor.py`)
Custom wrapper to support both attribute and dict access:

```python
class ConfigAccessor:
    def __init__(self, config: DachiConfig):
        self._config = config
        self._frozen = False

    def __getattr__(self, name: str):
        # config.Ops
        return getattr(self._config, name)

    def __getitem__(self, key: str):
        # config['Ops']
        return getattr(self._config, key)

    def freeze(self):
        self._frozen = True
```

#### 4. Global Singleton (`dachi/config/__init__.py`)
Lazy-initialized global config instance with support for custom config classes:

```python
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound='DachiConfig')

_global_config: ConfigAccessor | None = None
_config_class: Type[DachiConfig] = DachiConfig  # Default

def init_config(
    config_path: str | None = None,
    config_class: Type[T] | None = None
) -> ConfigAccessor:
    """Initialize global configuration.

    Args:
        config_path: Optional explicit path to config file
        config_class: Optional custom config class (must extend DachiConfig)

    Returns:
        ConfigAccessor wrapping the config instance
    """
    global _global_config, _config_class

    if _global_config is not None:
        raise RuntimeError("Config already initialized. Call reset_config() first.")

    # Use custom config class if provided
    if config_class is not None:
        _config_class = config_class

    # Load configuration
    if config_path:
        config_instance = _config_class(_env_file=None, yaml_file=config_path)
    else:
        config_instance = _config_class()

    # Wrap and freeze
    accessor = ConfigAccessor(config_instance)
    accessor.freeze()
    _global_config = accessor

    return accessor

def get_config() -> ConfigAccessor:
    """Get global configuration, initializing with defaults if needed."""
    global _global_config
    if _global_config is None:
        _global_config = init_config()
    return _global_config

def reset_config():
    """Reset global configuration (for testing)."""
    global _global_config, _config_class
    _global_config = None
    _config_class = DachiConfig  # Reset to default

# Module-level proxy
config = type('ConfigProxy', (), {
    '__getattr__': lambda self, name: getattr(get_config(), name),
    '__getitem__': lambda self, key: get_config()[key],
})()
```

**Key Features:**
- `init_config(config_class=MyCustomConfig)` allows users to extend config
- Lazy initialization: first access triggers loading
- Thread-safe singleton pattern
- Reset capability for testing

---

## 4. Integration with Existing Code

### 4.1 Operations (`dachi/proc/_operations.py`)

**Current Code:**
```python
DIFFERENCE_TEMPLATE = """Describe everything that is in Text A but is not in Text B.
Text A:
{a}
Text B:
{b}"""

def difference(a: str, b: str, _prompt: str=DIFFERENCE_TEMPLATE, _model=None):
    response, _, _ = Engines.get(_model).forward(
        prompt=str_formatter(_prompt, a=a, b=b)
    )
    return response
```

**New Code:**
```python
from dachi.config import config

def difference(a: str, b: str, _prompt: str | None = None, _model=None):
    # Use config defaults if not provided
    prompt = _prompt if _prompt is not None else config.Ops.Difference.prompt
    model = _model if _model is not None else config.Ops.Difference.model

    response, _, _ = Engines.get(model).forward(
        prompt=str_formatter(prompt, a=a, b=b)
    )
    return response
```

**Changes:**
- ✅ Remove global `DIFFERENCE_TEMPLATE` constant
- ✅ Change `_prompt` default from constant to `None`
- ✅ Add fallback: `_prompt if _prompt is not None else config.Ops.Difference.prompt`
- ✅ Add fallback: `_model if _model is not None else config.Ops.Difference.model`
- ✅ Maintains backward compatibility

**Apply to all functions:**
- `difference()`, `difference_stream()`, `async_difference()`, `difference_astream()`
- `union()`, `union_stream()`, `async_union()`, `union_astream()`
- `intersect()`, `intersect_stream()`, `async_intersect()`, `intersect_astream()`
- `symmetric_difference()`, etc.

---

### 4.2 Operation Classes

**Classes to update:**
- `Union` (lines 395-482 in `_operations.py`)
- `Intersection` (lines 485-560)
- `Difference` (lines 563-647)
- `SymmetricDifference` (lines 650-733)

**Pattern:**
```python
class Difference(LangEngine):
    sep: str = ""
    header: str = ""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        # Set defaults from config if not explicitly provided
        if self._model.data is None:
            self._model.data = config.Ops.Difference.model
```

---

### 4.3 Engines Registry (`dachi/proc/_lang.py`)

**Extend `Registry.get()` to use config default:**

**Current Code (line 355+):**
```python
Engines = Registry[LangModel]()
```

**New Code:**
```python
from dachi.config import config

class EngineRegistry(Registry[LangModel]):
    def get(self, key: str | None = None) -> LangModel:
        """Get engine, using config default if key is None."""
        if key is None:
            # Try config default first
            key = config.Engines.default_model

        if key is None:
            # Fall back to registry default
            if self._default is not None:
                return self._default.obj
            raise ValueError("No model specified and no default configured")

        if isinstance(key, str):
            return self._entries[key].obj

        return key

Engines = EngineRegistry()
```

**Changes:**
- ✅ Subclass `Registry[LangModel]` → `EngineRegistry`
- ✅ Override `get()` to check `config.Engines.default_model`
- ✅ Fall back to existing `self._default` if config not set
- ✅ Maintains backward compatibility

---

## 5. File Structure

```
dachi/
├── config/
│   ├── __init__.py          # Public API, global config singleton, exports 'config'
│   ├── _models.py           # Pydantic BaseSettings + nested BaseModel configs
│   └── _accessor.py         # Dual access wrapper (attribute + dict)
├── proc/
│   ├── _operations.py       # Modified: use config defaults
│   └── _lang.py             # Modified: EngineRegistry.get() uses config
├── __init__.py              # Auto-initialize config on import
└── py.typed                 # Type checking support marker
```

**Files to Create:**
- [`dachi/config/__init__.py`](dachi/config/__init__.py)
- [`dachi/config/_models.py`](dachi/config/_models.py)
- [`dachi/config/_accessor.py`](dachi/config/_accessor.py)

**Files to Modify:**
- [`dachi/proc/_operations.py`](dachi/proc/_operations.py) - ~15 functions, 4 classes
- [`dachi/proc/_lang.py`](dachi/proc/_lang.py) - `EngineRegistry` class
- [`dachi/__init__.py`](dachi/__init__.py) - Add lazy config init
- [`pyproject.toml`](pyproject.toml) - Add dependencies

**Test Files to Create:**
- [`tests/config/test_models.py`](tests/config/test_models.py)
- [`tests/config/test_loading.py`](tests/config/test_loading.py)
- [`tests/config/test_accessor.py`](tests/config/test_accessor.py)
- [`tests/config/test_integration.py`](tests/config/test_integration.py)

---

## 6. Example YAML Configuration

**File: `dachi.yaml`** (project root)

```yaml
# Dachi Framework Configuration
# See: https://dachi.readthedocs.io/config

# Text operations configuration
Ops:
  Difference:
    model: gpt-4
    prompt: |
      Describe everything that is in Text A but is not in Text B.
      Text A:
      {a}
      Text B:
      {b}

  SymmetricDifference:
    model: gpt-4
    prompt: |
      Describe everything that is in Text A but is not in Text B,
      and everything that is in Text B but is not in Text A.
      Text A:
      {a}
      Text B:
      {b}

  Union:
    model: claude-3-opus
    prompt: "Combine the following texts into one unified text, separating sections with '{sep}':\n{texts}"
    sep: "\n\n"
    header: "=== TEXT {} ===\n"

  Intersection:
    model: gpt-4
    prompt: "Given the following texts, extract and return only the content that is common to all texts.\nTexts:\n{texts}"
    sep: "\n"
    header: "**TEXT {}**\n"

# Processing configuration
Proc:
  AsyncDispatcher:
    max_concurrency: 10  # Increase from default 8 for high-throughput scenarios

  ToolUser:
    max_iterations: 15  # Allow more iterations for complex agentic tasks

  Parser:
    csv_delimiter: ','
    char_delimiter: ','
    line_separator: '\n'

# Behavior tree and state chart configuration
Act:
  StateChart:
    queue_maxsize: 2048  # Larger queue for high-frequency event systems
    queue_overflow: "drop_oldest"  # Drop old events instead of newest
    checkpoint_policy: "yield"
    emit_enforcement: "warn"
    auto_finish: true

  Parallel:
    fails_on: 2  # Require 2 failures before failing
    succeeds_on: 3  # Require 3 successes before succeeding
    success_priority: true
    preempt: false
    auto_run: true

# Instruction and criterion configuration
Inst:
  Criterion:
    likert_min: 1
    likert_max: 7  # Use 7-point Likert scale instead of default 5
    score_min: 0.0
    score_max: 100.0  # Use percentage scale (0-100) instead of 0-10
    membership_min: 0.0
    membership_max: 1.0

# Optimization configuration
Optim:
  LangOptim:
    prompt_template: |
      Optimize the following parameters.
      Objective: {objective}
      Constraints: {constraints}
      Current evaluations:
      {evaluations}

  LangCritic:
    prompt_template: |
      Evaluate the following output according to this criterion: {criterion}

      Output: {output}
      Input: {input}
      Reference: {reference}
      Context: {context}

# Default model for Engines registry
Engines:
  default_model: gpt-4
```

---

## 7. API Usage Examples

### Example 1: Basic Usage with Config Defaults
```python
import dachi
from dachi.proc import difference

# Uses model and prompt from config (or defaults if no config file)
result = difference("text a", "text b")
```

### Example 2: Override at Runtime
```python
# Config has gpt-4, but override to claude
result = difference("text a", "text b", _model="claude-3-opus")

# Override both model and prompt
result = difference(
    "text a",
    "text b",
    _model="gpt-4-turbo",
    _prompt="Custom prompt: {a} vs {b}"
)
```

### Example 3: Accessing Configuration
```python
from dachi.config import config

# Attribute access
model = config.Ops.Difference.model
prompt = config.Ops.Difference.prompt

# Dict access
model = config['Ops'].Difference.model
prompt = config['Ops']['Difference']['prompt']

# Mixed access
ops_config = config.Ops
diff_model = ops_config['Difference'].model
```

### Example 4: Explicit Initialization (Optional)
```python
from dachi.config import init_config

# Auto-discover config file in search paths
init_config()

# Or provide explicit path
init_config(config_path="/path/to/custom/config.yaml")
```

### Example 5: Environment Variable Override
```bash
# Terminal
export DACHI_OPS__DIFFERENCE__MODEL=gpt-4-turbo
export DACHI_ENGINES__DEFAULT_MODEL=claude-3
python my_script.py
```

```python
# In my_script.py
from dachi.config import config
print(config.Ops.Difference.model)  # "gpt-4-turbo" (from env)
```

### Example 6: User-Extensible Config (Custom Application Config)
```python
# In my_app/config.py
from dachi.config._models import DachiConfig
from pydantic import BaseModel, Field

class DatabaseConfig(BaseModel):
    url: str = "sqlite:///app.db"
    pool_size: int = 5
    echo: bool = False

class CacheConfig(BaseModel):
    backend: str = "redis"
    ttl: int = 3600
    max_size: int = 1000

class MyAppConfig(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    api_timeout: float = 30.0
    debug: bool = False

class AppConfig(DachiConfig):
    """Extended config with application-specific settings."""
    MyApp: MyAppConfig = Field(default_factory=MyAppConfig)
```

```python
# In my_app/main.py
from dachi.config import init_config
from my_app.config import AppConfig

# Initialize with custom config class
config = init_config(config_class=AppConfig)

# Access Dachi config (works as before)
print(config.Ops.Difference.model)

# Access custom application config
print(config.MyApp.database.url)
print(config.MyApp.cache.ttl)
```

```yaml
# dachi.yaml (supports both Dachi and custom config)
# Dachi config
Ops:
  Difference:
    model: gpt-4

Engines:
  default_model: gpt-4

# Custom application config
MyApp:
  database:
    url: "postgresql://localhost/production"
    pool_size: 20
    echo: false

  cache:
    backend: "memcached"
    ttl: 7200
    max_size: 5000

  api_timeout: 60.0
  debug: false
```

```bash
# Environment variables work for custom config too
export DACHI_MYAPP__DATABASE__URL=postgresql://prod-db/myapp
export DACHI_MYAPP__DEBUG=true
```

**Benefits:**
- Single config file for entire application
- Consistent YAML + env var pattern
- Type-safe access to custom settings
- No need to reinvent configuration system

---

## 8. Complete Inventory of Configurable Components

This section documents ALL hardcoded settings discovered in the Dachi codebase that should be made configurable.

### 8.1 Text Operations (`dachi/proc/_operations.py`)

| Component | Setting | Default Value | Type | Use Case |
|-----------|---------|---------------|------|----------|
| Difference | model | None | str\|None | LLM for difference operation |
| Difference | prompt | Template | str | Prompt for finding differences |
| SymmetricDifference | model | None | str\|None | LLM for symmetric diff |
| SymmetricDifference | prompt | Template | str | Prompt for symmetric diff |
| Union | model | None | str\|None | LLM for union operation |
| Union | prompt | Template | str | Prompt for combining texts |
| Union | sep | '\n' | str | Separator between texts |
| Union | header | "**TEXT {}**\n" | str | Header format for each text |
| Intersection | model | None | str\|None | LLM for intersection |
| Intersection | prompt | Template | str | Prompt for finding common content |
| Intersection | sep | '\n' | str | Separator between texts |
| Intersection | header | "**TEXT {}**\n" | str | Header format for each text |

**Files:** `_operations.py:7-734`

---

### 8.2 Processing Settings (`dachi/proc/`)

| Component | Setting | Default Value | Type | Use Case |
|-----------|---------|---------------|------|----------|
| AsyncDispatcher | max_concurrency | 8 | int | Max concurrent API requests |
| ToolUser | max_iterations | 10 | int | Max agent loop iterations |
| CSVRowParser | delimiter | ',' | str | CSV field delimiter |
| CharDelimParser | sep | ',' | str | Character delimiter |
| LineSepParser | sep | '\n' | str | Line separator |

**Files:**
- `_dispatch.py:126, 187` (AsyncDispatcher)
- `_lang.py:536` (ToolUser)
- `_parser.py:75, 199, 290` (Parsers)

**Rationale:**
- Different cloud providers have different rate limits
- Complex agentic tasks may need more/fewer iterations
- Data sources use different delimiters

---

### 8.3 Behavior Trees & State Charts (`dachi/act/`)

#### State Chart Configuration

| Component | Setting | Default Value | Type | Use Case |
|-----------|---------|---------------|------|----------|
| StateChart | queue_maxsize | 1024 | int | Event queue capacity |
| StateChart | queue_overflow | "drop_newest" | Literal | Overflow handling strategy |
| StateChart | checkpoint_policy | "yield" | Literal | Checkpoint mode |
| StateChart | emit_enforcement | "warn" | Literal | Event validation strictness |
| StateChart | auto_finish | True | bool | Auto-finalization behavior |

**Files:** `chart/_chart.py:34-38`, `chart/_event.py:50-51`

#### Parallel Task Configuration

| Component | Setting | Default Value | Type | Use Case |
|-----------|---------|---------------|------|----------|
| MultiTask | fails_on | 1 | int | Failure threshold |
| MultiTask | succeeds_on | -1 | int | Success threshold (-1 = all) |
| MultiTask | success_priority | True | bool | Success over failure priority |
| MultiTask | preempt | False | bool | Preempt on failure |
| MultiTask | auto_run | True | bool | Auto-execute tasks |

**Files:** `bt/_parallel.py:26-30`

**Rationale:**
- High-frequency event systems need larger queues
- Different parallel patterns need different completion criteria
- Production systems may want "drop_oldest" to prioritize recent events

---

### 8.4 Instruction & Criterion Settings (`dachi/inst/`)

| Component | Setting | Default Value | Type | Use Case |
|-----------|---------|---------------|------|----------|
| Criterion | likert_min | 1 | int | Minimum Likert scale rating |
| Criterion | likert_max | 5 | int | Maximum Likert scale rating |
| Criterion | score_min | 0.0 | float | Minimum numerical score |
| Criterion | score_max | 10.0 | float | Maximum numerical score |
| Criterion | membership_min | 0.0 | float | Minimum fuzzy membership |
| Criterion | membership_max | 1.0 | float | Maximum fuzzy membership |

**Files:** `_criterion.py:41, 49, 74, 88, 123`

**Rationale:**
- Different evaluation contexts use different scales
- Some applications use 7-point Likert, others 5-point
- Scores might be 0-100 (percentage) vs 0-10

---

### 8.5 Optimization Settings (`dachi/proc/_optim.py`)

| Component | Setting | Default Value | Type | Use Case |
|-----------|---------|---------------|------|----------|
| LangOptim | prompt_template | Template | str | Optimization prompt |
| LangCritic | prompt_template | Template | str | Evaluation prompt |

**Files:** (To be explored in _optim.py)

---

### 8.6 Engine Registry (`dachi/proc/_lang.py`)

| Component | Setting | Default Value | Type | Use Case |
|-----------|---------|---------------|------|----------|
| Engines | default_model | None | str\|None | Default LLM model key |

**Files:** `_lang.py:355`

**Rationale:** Allows setting project-wide default model

---

### Summary Table: Config Namespace Organization

| Namespace | # Settings | Files Affected | Priority |
|-----------|-----------|----------------|----------|
| **Ops** | 12 | _operations.py | High |
| **Proc** | 8 | _dispatch.py, _lang.py, _parser.py | High |
| **Act** | 10 | chart/_chart.py, bt/_parallel.py | Medium |
| **Inst** | 6 | _criterion.py | Medium |
| **Optim** | 2 | _optim.py | High |
| **Engines** | 1 | _lang.py | High |
| **TOTAL** | 39 settings | 9 files | - |

---

## 9. Dependencies

### New Required Dependencies

Add to `pyproject.toml`:

```toml
[tool.poetry.dependencies]
pydantic-settings = "^2.0"  # Official Pydantic settings extension
pyyaml = "^6.0"             # YAML parsing (required by pydantic-settings)
```

### Existing Dependencies (Already in Project)
- `pydantic = ">=2, <3"` ✅ Already present

**Installation:**
```bash
poetry add pydantic-settings pyyaml
```

---

## 9. Testing Strategy

### Test Coverage Requirements
- **Unit tests:** >90% coverage for config module
- **Integration tests:** Verify operations use config correctly
- **End-to-end tests:** Full precedence chain (defaults → YAML → env)

### Test File Structure

```
tests/
└── config/
    ├── test_models.py         # Pydantic model validation
    ├── test_loading.py        # YAML + env var loading
    ├── test_accessor.py       # Dual access pattern
    └── test_integration.py    # Integration with operations
```

---

### Test Cases

#### `test_models.py` - Model Validation
```python
class TestDifferenceConfig:
    def test_default_values_set(self):
        """Defaults from Field() are populated."""
        config = DifferenceConfig()
        assert config.model is None
        assert "Text A" in config.prompt

    def test_custom_model_accepted(self):
        """Can override default model."""
        config = DifferenceConfig(model="gpt-4")
        assert config.model == "gpt-4"

    def test_invalid_type_rejected(self):
        """Pydantic validates types."""
        with pytest.raises(ValidationError):
            DifferenceConfig(model=123)  # Should be str or None


class TestDachiConfig:
    def test_nested_structure(self):
        """Nested models work."""
        config = DachiConfig()
        assert isinstance(config.Ops, OpsConfig)
        assert isinstance(config.Ops.Difference, DifferenceConfig)
```

---

#### `test_loading.py` - YAML and Env Loading
```python
class TestYamlLoading:
    def test_load_from_yaml(self, tmp_path):
        """YAML file loads correctly."""
        yaml_file = tmp_path / "dachi.yaml"
        yaml_file.write_text("""
Ops:
  Difference:
    model: gpt-4
""")

        config = DachiConfig(_env_file=None, yaml_file=str(yaml_file))
        assert config.Ops.Difference.model == "gpt-4"

    def test_missing_yaml_uses_defaults(self):
        """No YAML file uses Field defaults."""
        config = DachiConfig(_env_file=None)
        assert config.Ops.Difference.model is None
        assert "Text A" in config.Ops.Difference.prompt


class TestEnvVarLoading:
    def test_env_var_override(self, monkeypatch):
        """Environment variables override YAML."""
        monkeypatch.setenv("DACHI_OPS__DIFFERENCE__MODEL", "env-model")

        config = DachiConfig(_env_file=None)
        assert config.Ops.Difference.model == "env-model"

    def test_precedence_order(self, tmp_path, monkeypatch):
        """Env > YAML > defaults."""
        yaml_file = tmp_path / "dachi.yaml"
        yaml_file.write_text("Ops:\n  Difference:\n    model: yaml-model")

        monkeypatch.setenv("DACHI_OPS__DIFFERENCE__MODEL", "env-model")

        config = DachiConfig(_env_file=None, yaml_file=str(yaml_file))
        assert config.Ops.Difference.model == "env-model"
```

---

#### `test_accessor.py` - Dual Access Pattern
```python
class TestConfigAccessor:
    def test_attribute_access(self):
        """config.Ops.Difference.model works."""
        config = ConfigAccessor(DachiConfig(_env_file=None))
        assert config.Ops.Difference.model is None

    def test_dict_access(self):
        """config['Ops'].Difference.model works."""
        config = ConfigAccessor(DachiConfig(_env_file=None))
        assert config['Ops'].Difference.model is None

    def test_full_dict_access(self):
        """config['Ops']['Difference']['model'] works."""
        config = ConfigAccessor(DachiConfig(_env_file=None))
        assert config['Ops']['Difference']['model'] is None

    def test_freeze_prevents_modification(self):
        """Frozen config is immutable."""
        config = ConfigAccessor(DachiConfig(_env_file=None))
        config.freeze()

        with pytest.raises(RuntimeError, match="read-only"):
            config.Ops.Difference.model = "new"
```

---

#### `test_integration.py` - Integration with Operations
```python
class TestOperationsIntegration:
    def test_difference_uses_config_default(self, monkeypatch):
        """difference() uses config.Ops.Difference.model."""
        monkeypatch.setenv("DACHI_OPS__DIFFERENCE__MODEL", "test-model")
        # Reset global config to reload with env var
        reset_config()

        # Mock Engines.get() to verify it receives "test-model"
        with patch('dachi.proc._operations.Engines.get') as mock_get:
            mock_model = MagicMock()
            mock_model.forward.return_value = ("result", None, None)
            mock_get.return_value = mock_model

            difference("a", "b")

            # Verify Engines.get was called with config default
            mock_get.assert_called_once_with("test-model")

    def test_explicit_param_overrides_config(self):
        """_model parameter overrides config."""
        with patch('dachi.proc._operations.Engines.get') as mock_get:
            mock_model = MagicMock()
            mock_model.forward.return_value = ("result", None, None)
            mock_get.return_value = mock_model

            difference("a", "b", _model="override-model")

            mock_get.assert_called_once_with("override-model")
```

---

## 10. Migration and Implementation Plan

### Phase 1: Setup Config Module
**Goal:** Create config package with pydantic-settings

**Tasks:**
1. Create `dachi/config/` directory
2. Add dependencies to `pyproject.toml`:
   - `pydantic-settings = "^2.0"`
   - `pyyaml = "^6.0"`
3. Run `poetry install`
4. Implement `dachi/config/_models.py`:
   - All config classes (DifferenceConfig, UnionConfig, etc.)
   - DachiConfig(BaseSettings) root with YamlConfigSettingsSource
5. Implement `dachi/config/_accessor.py`:
   - ConfigAccessor with dual access pattern
   - Freeze functionality
6. Implement `dachi/config/__init__.py`:
   - Global singleton
   - init_config(), get_config(), reset_config()
   - Export `config` proxy

**Verification:**
```bash
pytest tests/config/test_models.py -v
pytest tests/config/test_accessor.py -v
```

---

### Phase 2: Add Tests
**Goal:** Comprehensive test coverage before integration

**Tasks:**
1. Create `tests/config/` directory
2. Implement `test_models.py` - validation tests
3. Implement `test_loading.py` - YAML + env var tests
4. Implement `test_accessor.py` - dual access tests
5. Ensure >90% coverage

**Verification:**
```bash
pytest tests/config/ -v --cov=dachi/config --cov-report=term-missing
# Expect >90% coverage
```

---

### Phase 3: Integrate with Operations
**Goal:** Modify `_operations.py` to use config defaults

**Tasks:**
1. Remove global template constants:
   - `DIFFERENCE_TEMPLATE`
   - `SYMMETRIC_DIFFERENCE_TEMPLATE`
   - `UNION_TEMPLATE`
   - `INTERSECTION_TEMPLATE`
2. Update function signatures (~15 functions):
   - Change `_prompt=TEMPLATE` → `_prompt=None`
   - Add fallback logic: `_prompt if _prompt is not None else config.Ops.<Op>.prompt`
   - Add model fallback: `_model if _model is not None else config.Ops.<Op>.model`
3. Update operation classes (Union, Difference, Intersection, SymmetricDifference):
   - Add config defaults in `model_post_init()`
4. Add integration tests in `test_integration.py`

**Verification:**
```bash
pytest tests/config/test_integration.py -v
pytest tests/proc/test_operations.py -v  # Existing tests still pass
```

---

### Phase 4: Integrate with Engines Registry
**Goal:** Engines.get() uses config default

**Tasks:**
1. Modify `dachi/proc/_lang.py`:
   - Create `EngineRegistry` subclass of `Registry[LangModel]`
   - Override `get()` to check `config.Engines.default_model`
   - Replace `Engines = Registry[LangModel]()` → `Engines = EngineRegistry()`
2. Add tests for default model lookup

**Verification:**
```bash
pytest tests/proc/test_lang.py -v  # Test Engines.get() behavior
```

---

### Phase 5: Auto-Initialize and Documentation
**Goal:** Lazy config loading and user docs

**Tasks:**
1. Modify `dachi/__init__.py`:
   - Add lazy config initialization
   - Ensure `dachi.config` is accessible
2. Create example `dachi.yaml` in repo root with comments
3. Update `README.md`:
   - Add "Configuration" section with examples
   - Link to docs
4. Add Sphinx documentation:
   - `docs/configuration.rst` - comprehensive guide
   - API reference for config module

**Verification:**
```bash
# Test import
python -c "import dachi; print(dachi.config.Ops.Difference.model)"

# Build docs
cd docs && make html
```

---

### Phase 6: End-to-End Verification
**Goal:** Full system test

**Tasks:**
1. Create test project with `dachi.yaml`
2. Set environment variables
3. Run operations and verify precedence
4. Run full test suite
5. Check documentation builds

**Verification:**
```bash
# Full test suite
pytest tests/ -v

# Check coverage
pytest --cov=dachi --cov-report=html

# Build docs
cd docs && make clean && make html
```

---

## 11. Security Considerations

### 11.1 API Keys and Secrets

**Problem:** Config files may contain sensitive data (API keys, credentials)

**Solutions:**

#### Option 1: Environment Variables Only (Recommended)
```bash
# Never commit API keys to YAML
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# In dachi.yaml - only non-sensitive config
Engines:
  default_model: gpt-4  # OK - not a secret
```

**Benefits:**
- ✅ Secrets never in version control
- ✅ Standard deployment practice
- ✅ Works with Docker, K8s, cloud platforms

#### Option 2: Separate Secrets File
```yaml
# dachi.yaml (committed)
Ops:
  Difference:
    model: gpt-4

# secrets.yaml (gitignored, not committed)
api_keys:
  openai: sk-...
  anthropic: sk-ant-...
```

Add to `.gitignore`:
```
secrets.yaml
.env
*.secret.yaml
```

#### Option 3: Use Existing .env File
The existing `.env` file at project root already contains API keys. Recommendation:
- **Keep API keys in `.env` only** (already gitignored)
- **Use config (YAML/env vars) for non-sensitive settings only**
- Document in README: "API keys go in `.env`, config settings go in `dachi.yaml`"

### 11.2 YAML File Permissions

**Recommendation:**
If storing any sensitive data in YAML, set restrictive permissions:

```bash
chmod 600 dachi.yaml  # Owner read/write only
```

### 11.3 Security Best Practices

1. **Never commit secrets** - Use `.gitignore` for `*.secret.yaml`, `.env`
2. **Validate YAML safely** - pydantic-settings uses `yaml.safe_load()` (prevents code execution)
3. **Audit config loading** - Log warnings if sensitive fields detected in YAML:
   ```python
   if "api_key" in config_data or "secret" in config_data:
       warnings.warn("Sensitive data detected in config file. Use env vars instead.")
   ```
4. **Document security** - Add security section to docs explaining separation of concerns

---

## 12. Performance Considerations

### 12.1 Config Loading Performance

**Lazy Loading Strategy (Recommended):**
```python
# In dachi/__init__.py
_config = None

def get_config():
    global _config
    if _config is None:
        _config = _load_config()  # Only load once on first access
    return _config
```

**Performance Characteristics:**
- **First load:** ~5-20ms (parse YAML, validate with Pydantic)
- **Subsequent access:** ~0.001ms (in-memory singleton)
- **Import time:** 0ms (lazy init means no overhead on `import dachi`)

### 12.2 YAML Parsing Optimization

**Small config files (<100KB):** No optimization needed

**Large config files (>1MB):**
```python
# Use ruamel.yaml or faster YAML parser if needed
from ruamel.yaml import YAML
yaml = YAML()
yaml.default_flow_style = False
```

**Recommendation:** Start with `pyyaml.safe_load()`, optimize only if profiling shows bottleneck.

### 12.3 Environment Variable Parsing

pydantic-settings parses env vars efficiently:
- **Single-level:** `DACHI_FIELD` → O(1) lookup
- **Nested:** `DACHI_OPS__DIFFERENCE__MODEL` → O(depth) parsing
- **Impact:** Negligible (<1ms for typical 10-50 env vars)

### 12.4 Memory Footprint

**Typical config object:**
- Config schema: ~5-10 KB
- YAML data: ~1-50 KB
- Total: <100 KB in memory

**Optimization:** Config is loaded once and frozen (immutable), so no memory growth.

### 12.5 Caching Strategy

**No additional caching needed:**
- Singleton pattern = natural cache
- Config frozen after load = safe to share across threads/async
- No invalidation logic needed (no hot reloading)

### 12.6 Performance Testing

Add performance test:
```python
# tests/config/test_performance.py
def test_config_load_time():
    """Config loads in <50ms."""
    import time
    start = time.time()
    config = DachiConfig()
    elapsed = time.time() - start
    assert elapsed < 0.05  # 50ms threshold

def test_config_access_time():
    """Config access is fast."""
    config = get_config()
    start = time.time()
    for _ in range(10000):
        _ = config.Ops.Difference.model
    elapsed = time.time() - start
    assert elapsed < 0.01  # 10ms for 10k accesses
```

---

## 13. Error Handling and Debugging

### 13.1 Error Types and Messages

#### Error 1: Invalid YAML Syntax
**Cause:** Malformed YAML file

**Example:**
```yaml
Ops:
  Difference:
    model gpt-4  # Missing colon
```

**Error Message:**
```
ValueError: Invalid YAML in /path/to/dachi.yaml:
  yaml.scanner.ScannerError: mapping values are not allowed here
  in "dachi.yaml", line 3, column 11
```

**Solution:**
- Show exact line and column
- Include snippet of problematic YAML
- Suggest: "Check YAML syntax at https://www.yamllint.com/"

#### Error 2: Validation Failure
**Cause:** Config value has wrong type

**Example:**
```yaml
Ops:
  Difference:
    model: 123  # Should be string or null
```

**Error Message:**
```
pydantic_core._pydantic_core.ValidationError: 2 validation errors for DachiConfig
Ops.Difference.model
  Input should be a valid string [type=string_type, input_value=123, input_type=int]
    For further information visit https://errors.pydantic.dev/2.0/v/string_type
```

**Enhanced Error:**
```python
try:
    config = DachiConfig()
except ValidationError as e:
    # Pretty-print with context
    print(f"Configuration validation failed in {yaml_path}:")
    for error in e.errors():
        field = '.'.join(str(loc) for loc in error['loc'])
        print(f"  {field}: {error['msg']}")
        print(f"    Got: {error['input']} (type: {type(error['input']).__name__})")
    raise
```

#### Error 3: Missing Required Field
**Cause:** Required field not provided (if we add any)

**Currently:** All fields have defaults, so this won't happen

**Future:** If adding required fields:
```python
class DifferenceConfig(BaseModel):
    model: str  # No default = required
```

**Error Message:**
```
ValidationError: 1 validation error for DifferenceConfig
model
  Field required [type=missing]
```

#### Error 4: File Not Found (Explicit Path)
**Cause:** `init_config(config_path="/nonexistent.yaml")`

**Error Message:**
```python
raise FileNotFoundError(
    f"Config file not found: {config_path}\n"
    f"Searched paths: {searched_paths}\n"
    f"Tip: Create 'dachi.yaml' in your project root or use default search paths."
)
```

#### Error 5: Model Not in Registry
**Cause:** `config.Ops.Difference.model = "unknown-model"`

**Error:** Happens when calling operation, not at config load
```python
# In Engines.get()
if key not in self._entries:
    raise KeyError(
        f"Model '{key}' not found in Engines registry.\n"
        f"Available models: {list(self._entries.keys())}\n"
        f"Register models using: Engines.register(<name>, <model_instance>)"
    )
```

### 13.2 Debugging Features

#### Debug Mode
```python
# In dachi/config/__init__.py
import os

DEBUG = os.getenv("DACHI_DEBUG", "").lower() in ("1", "true", "yes")

def get_config():
    if DEBUG:
        print(f"[DEBUG] Loading config from: {search_paths}")
        print(f"[DEBUG] Environment variables: {env_vars}")

    config = DachiConfig()

    if DEBUG:
        print(f"[DEBUG] Loaded config: {config.model_dump()}")

    return config
```

**Usage:**
```bash
export DACHI_DEBUG=1
python script.py
# [DEBUG] Loading config from: ['./dachi.yaml', ...]
# [DEBUG] Loaded config: {'Ops': {'Difference': {'model': 'gpt-4', ...}}}
```

#### Config Validation CLI
Create optional CLI tool for validating config:

```python
# dachi/config/_cli.py
import sys
from pathlib import Path
from pydantic import ValidationError

def validate_config(path: str):
    """Validate config file without loading Dachi."""
    try:
        config = DachiConfig(yaml_file=path)
        print(f"✓ Config valid: {path}")
        print("\nLoaded configuration:")
        print(config.model_dump_json(indent=2))
        return 0
    except ValidationError as e:
        print(f"✗ Config invalid: {path}")
        print("\nErrors:")
        for error in e.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            print(f"  {field}: {error['msg']}")
        return 1
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(validate_config(sys.argv[1] if len(sys.argv) > 1 else "dachi.yaml"))
```

**Usage:**
```bash
python -m dachi.config._cli dachi.yaml
# ✓ Config valid: dachi.yaml
# Loaded configuration: {...}
```

### 13.3 Troubleshooting Guide

**Issue:** "Config not loading from YAML"

**Debug steps:**
1. Check file exists: `ls -la dachi.yaml`
2. Check file permissions: `ls -l dachi.yaml` (should be readable)
3. Validate YAML syntax: `python -m dachi.config._cli dachi.yaml`
4. Enable debug mode: `DACHI_DEBUG=1 python script.py`
5. Check search paths: Print `ConfigLoader.SEARCH_PATHS`

**Issue:** "Environment variables not overriding YAML"

**Debug steps:**
1. Check env var name: Must be `DACHI_<NAMESPACE>__<CLASS>__<FIELD>` (double underscore)
2. Check case: pydantic-settings is case-insensitive, but use UPPERCASE for clarity
3. Verify env var is set: `echo $DACHI_OPS__DIFFERENCE__MODEL`
4. Enable debug mode to see precedence

**Issue:** "Config changes not taking effect"

**Cause:** Config is frozen and cached

**Solution:**
1. Restart Python process (no hot reload)
2. For tests: Use `reset_config()` fixture
3. For notebooks: Restart kernel

### 13.4 Logging

Add structured logging:
```python
import logging

logger = logging.getLogger("dachi.config")

def _load_config():
    logger.info("Loading Dachi configuration")

    yaml_path = _find_yaml()
    if yaml_path:
        logger.info(f"Found config file: {yaml_path}")
    else:
        logger.info("No config file found, using defaults")

    env_vars = {k: v for k, v in os.environ.items() if k.startswith("DACHI_")}
    if env_vars:
        logger.info(f"Found {len(env_vars)} DACHI_* environment variables")

    config = DachiConfig()
    logger.info("Configuration loaded successfully")

    return config
```

---

## 14. Constraints and Assumptions

### Constraints
1. **Initialization-only:** Config is immutable after first load (frozen)
2. **Single global instance:** One config per process (singleton pattern)
3. **No runtime reloading:** Changes to YAML or env vars require process restart
4. **Backward compatible:** Existing code with explicit parameters continues to work
5. **Python 3.10+:** Uses modern type hints (`str | None`)
6. **No secrets in config:** API keys must go in `.env` or env vars, not YAML

### Assumptions
1. Users run Dachi from a project directory or have `~/.dachi/` setup
2. YAML syntax is familiar to target users
3. Environment variable override is sufficient for deployment flexibility
4. Config values are simple types (str, int, bool, None) - no complex objects
5. pydantic-settings is actively maintained (it's official Pydantic extension)
6. Config files are small (<1MB), performance is not a concern

---

## 15. Success Criteria

**The implementation is successful when:**

### Core Functionality
1. ✅ Users can create `dachi.yaml` and configure default models/prompts
2. ✅ Environment variables override YAML values (precedence: env > YAML > defaults)
3. ✅ Both `config.Ops.Difference.model` and `config['Ops'].Difference.model` work
4. ✅ Invalid config raises clear Pydantic validation errors with field names
5. ✅ Config is frozen after initialization (prevents accidental modification)

### Integration
6. ✅ Operations (`difference`, `union`, etc.) use config defaults when parameters not provided
7. ✅ Function parameters (`_model`, `_prompt`) override config (backward compatible)
8. ✅ All 39 configurable settings from Section 8 are accessible via config
9. ✅ `AsyncDispatcher.max_concurrency` can be set via config
10. ✅ `ToolUser.max_iterations` can be set via config
11. ✅ State chart and behavior tree settings configurable

### Extensibility
12. ✅ Application config can remain separate via composition with its own env prefix and YAML
13. ✅ `init_config(config_class=MyCustomConfig)` still works for single-config setups
14. ✅ Custom config supports YAML and environment variables in both patterns
15. ✅ Framework contributors can add new namespaces (Story 6)

### Quality
16. ✅ All tests pass with >90% coverage for config module
17. ✅ Documentation explains configuration clearly with examples
18. ✅ User extensibility documented with working examples
19. ✅ Performance meets targets (load <50ms, access <0.001ms)
20. ✅ Security best practices documented (secrets in .env, not YAML)

---

## 16. Open Questions

### Question 1: Should we support multiple config files?
**Example:** `dachi.base.yaml` + `dachi.local.yaml` merged together

**Options:**
- **A) Single file only** - Simpler, explicit
- **B) Multiple files with merge** - More flexible for team environments

**Recommendation:** Start with A, add B later if needed

---

### Question 2: Should config support computed/derived values?
**Example:** `Ops.Difference.model` defaults to `Engines.default_model` if not set

**Options:**
- **A) Static values only** - Simpler, explicit
- **B) Support references** - More DRY, less duplication

**Recommendation:** Start with A (static), users can set both explicitly

---

### Question 3: Should we validate model names against Engines registry?
**Example:** Raise error if `config.Ops.Difference.model = "unknown-model"` not in registry

**Options:**
- **A) No validation** - Config independent of runtime state
- **B) Lazy validation** - Validate when actually used
- **C) Strict validation** - Validate at config load time

**Recommendation:** A or B (C requires Engines to be registered before config loads)

---

## 17. Future Extensions (Out of Scope)

### Potential Future Features
- **Config profiles:** Multiple named configs (dev, staging, prod)
- **Multiple file merge:** `dachi.base.yaml` + `dachi.local.yaml`
- **Config validation CLI:** Tool to validate YAML before running
- **Config generation:** Generate template YAML from current defaults
- **Remote config:** Load config from URLs or cloud storage
- **Secrets management:** Integration with vaults for API keys
- **TOML support:** Use `pyproject.toml` for config instead of separate file
- **JSON schema export:** Generate JSON schema for IDE validation

### Extension Points
The design supports future extensions:
- **New namespaces:** Add new `BaseModel` to `DachiConfig`
- **New config sources:** Override `settings_customise_sources()` in `DachiConfig`
- **New access patterns:** Extend `ConfigAccessor` methods
- **Validation hooks:** Override Pydantic `@field_validator` or `@model_validator`

---

## 18. References and Resources

### Pydantic Settings Documentation
- [Settings Management - Pydantic Validation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Pydantic Settings API](https://docs.pydantic.dev/latest/api/pydantic_settings/)
- [Settings Management (v1.10)](https://docs.pydantic.dev/1.10/usage/settings/)

### YAML Configuration Examples
- [How to Load Configuration in Pydantic](https://medium.com/@wihlarkop/how-to-load-configuration-in-pydantic-3693d0ee81a3)
- [A simple guide to configure your Python project with Pydantic and a YAML file](https://medium.com/@jonathan_b/a-simple-guide-to-configure-your-python-project-with-pydantic-and-a-yaml-file-bef76888f366)
- [Keeping Configurations Sane with Pydantic Settings](https://ai.ragv.in/posts/sane-configs-with-pydantic-settings/)

### Related Libraries
- [pydantic-settings-yaml · PyPI](https://pypi.org/project/pydantic-settings-yaml/) - Alternative YAML loader
- [yaml-settings-pydantic · PyPI](https://pypi.org/project/yaml-settings-pydantic/) - Another YAML integration
- [FastAPI Settings](https://fastapi.tiangolo.com/advanced/settings/) - Example usage in production framework

---

## 19. Critical Files Reference

### Files to Create
- [`dachi/config/__init__.py`](dachi/config/__init__.py) - Public API, global singleton, exports `config`
- [`dachi/config/_models.py`](dachi/config/_models.py) - Pydantic BaseSettings + nested configs
- [`dachi/config/_accessor.py`](dachi/config/_accessor.py) - Dual access wrapper (attribute + dict)
- [`tests/config/test_models.py`](tests/config/test_models.py) - Model validation tests
- [`tests/config/test_loading.py`](tests/config/test_loading.py) - YAML + env var loading tests
- [`tests/config/test_accessor.py`](tests/config/test_accessor.py) - Dual access tests
- [`tests/config/test_integration.py`](tests/config/test_integration.py) - Integration with operations

### Files to Modify
- [`dachi/proc/_operations.py`](dachi/proc/_operations.py) - Use config defaults (~15 functions, 4 classes)
- [`dachi/proc/_lang.py`](dachi/proc/_lang.py) - EngineRegistry.get() uses config default
- [`dachi/__init__.py`](dachi/__init__.py) - Auto-initialize config on import
- [`pyproject.toml`](pyproject.toml) - Add pydantic-settings, pyyaml dependencies
- [`README.md`](README.md) - Add configuration section
- [`docs/index.rst`](docs/index.rst) - Link to configuration guide

### Files to Create (Documentation)
- `dachi.yaml` - Example configuration file in repo root
- `docs/configuration.rst` - Comprehensive configuration guide

---

**End of Specification**
