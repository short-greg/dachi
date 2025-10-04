# Testing Guide

This directory contains tests for the Dachi AI framework organized by test type.

## Test Organization

```
tests/
├── act/                    # Unit tests for behavior trees and state charts
├── core/                   # Unit tests for core framework components
├── proc/                   # Unit tests for processing components
├── inst/                   # Unit tests for instruction components
├── integration/            # Integration tests (opt-in)
├── e2e/                    # End-to-end tests (opt-in)
└── README.md              # This file
```

## Test Types

### Unit Tests (Default)
**Location**: `tests/act/`, `tests/core/`, `tests/proc/`, `tests/inst/`

**Purpose**: Fast, isolated tests that verify individual components work correctly.

**Characteristics**:
- Run quickly (< 100ms per test typically)
- No external dependencies
- Mock/stub dependencies
- Always run in CI/CD

**Run unit tests**:
```bash
# Run all unit tests (default)
pytest tests/act/
pytest tests/core/

# Run specific test file
pytest tests/act/test_chart.py

# Run with verbose output
pytest tests/act/ -v
```

### Integration Tests (Opt-in)
**Location**: `tests/integration/`

**Purpose**: Verify multiple components work together correctly.

**Characteristics**:
- Slower than unit tests (may take several seconds)
- Test real interactions between components
- Verify data flows through the system
- Run selectively (pre-commit, nightly builds)

**Scenarios Covered**:
- Multi-state workflows with event transitions
- Concurrent region coordination
- Preemption and cancellation flows
- Timer integration with state lifecycle
- Event queue stress testing
- State lifecycle (enter/exit/execute) ordering

**Run integration tests**:
```bash
# Run only integration tests
pytest tests/integration/ -m integration

# Run unit + integration tests
pytest tests/ -m "unit or integration"

# Run integration tests with verbose output
pytest tests/integration/ -v -m integration
```

### End-to-End Tests (Opt-in)
**Location**: `tests/e2e/`

**Purpose**: Test complete, realistic usage scenarios from start to finish.

**Characteristics**:
- Slowest tests (may take multiple seconds)
- Simulate real-world use cases
- Test the entire system working together
- Run manually or in nightly builds

**Scenarios Covered**:
1. **Multi-step wizard workflow**: Form with validation and payment processing
2. **Request-response with timeout**: HTTP-like request with retry logic
3. **Parallel task coordination**: Multiple concurrent tasks with synchronization
4. **Background job cancellation**: Long-running job with monitoring and cancellation
5. **Complex state machine**: Document editor with editing, saving, and error handling

**Run e2e tests**:
```bash
# Run only e2e tests
pytest tests/e2e/ -m e2e

# Run e2e tests with detailed output
pytest tests/e2e/ -v -m e2e -s

# Run specific e2e scenario
pytest tests/e2e/test_chart_e2e.py::TestWizardWorkflow -v -m e2e
```

## Test Markers

Tests are marked with pytest markers to enable selective execution:

- `@pytest.mark.unit`: Unit tests (implicit, don't need to add)
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.e2e`: End-to-end tests
- `@pytest.mark.slow`: Tests taking > 1 second

## Running Different Test Combinations

```bash
# Default: Run only unit tests (fast)
pytest tests/act/

# Run unit and integration tests
pytest tests/ -m "unit or integration"

# Run all tests including e2e
pytest tests/ -m "unit or integration or e2e"

# Run only slow tests
pytest tests/ -m slow

# Run everything except slow tests
pytest tests/ -m "not slow"

# Run with coverage
pytest tests/act/ --cov=dachi.act --cov-report=html
```

## CI/CD Recommendations

**Pull Request Checks** (fast feedback):
```bash
pytest tests/act/ tests/core/ tests/proc/ tests/inst/
```

**Pre-merge Checks** (more thorough):
```bash
pytest tests/ -m "unit or integration"
```

**Nightly Builds** (comprehensive):
```bash
pytest tests/ -m "unit or integration or e2e"
```

**Release Validation** (everything):
```bash
pytest tests/ tests_adapt/ -m "unit or integration or e2e"
```

## Writing New Tests

### Unit Test Template
```python
# tests/act/test_component.py
import pytest
from dachi.act import Component

class TestComponent:
    def test_component_initializes(self):
        component = Component()
        assert component is not None
```

### Integration Test Template
```python
# tests/integration/test_feature_integration.py
import pytest
import asyncio

pytestmark = pytest.mark.integration

class TestFeatureIntegration:
    @pytest.mark.asyncio
    async def test_components_work_together(self):
        # Setup multiple components
        # Exercise interactions
        # Verify integrated behavior
        pass
```

### E2E Test Template
```python
# tests/e2e/test_scenario_e2e.py
import pytest
import asyncio

pytestmark = pytest.mark.e2e

class TestRealisticScenario:
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        # Simulate real-world usage
        # Run complete scenario start to finish
        # Verify end state
        pass
```

## Debugging Failed Tests

**Run single test with output**:
```bash
pytest tests/act/test_chart.py::TestChartLifecycle::test_chart_initializes_in_idle -v -s
```

**Run with debugger**:
```bash
pytest tests/act/test_chart.py --pdb
```

**Show local variables on failure**:
```bash
pytest tests/act/ -v -l
```

**Stop on first failure**:
```bash
pytest tests/act/ -x
```

## Performance Tips

- Run unit tests frequently during development (fast feedback)
- Run integration tests before committing (catch integration issues)
- Run e2e tests before releases (validate complete scenarios)
- Use `-n auto` with pytest-xdist for parallel execution:
  ```bash
  pytest tests/ -n auto
  ```
