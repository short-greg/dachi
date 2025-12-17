# Criterion System

The Criterion system provides structured evaluation schemas for LLM-based assessments. Rather than free-form text feedback, criteria define exact fields and types that LLMs must populate, enabling consistent, automated evaluation workflows.

## Overview

**ResponseSpec** is the base class for creating evaluation schemas. It automatically generates Pydantic models from field descriptors, ensuring LLM responses conform to your evaluation structure.

Key Components:
- **ResponseSpec**: Base class for all criteria
- **RespField**: Field descriptors (TextField, BoolField, BoundInt, etc.)
- **Built-in Criteria**: Pre-defined evaluation types
- **Generated Schemas**: Auto-generated Pydantic models for LLM responses

## Core Concepts

### ResponseSpec

A ResponseSpec defines the structure of an evaluation:

```python
from dachi.inst import PassFailCriterion, BoolField, TextField

# Create a criterion
criterion = PassFailCriterion(
    name="safety_check",
    description="Evaluate content for safety",
    passed=BoolField(description="Whether content is safe"),
    passing_criteria=TextField(description="What made it safe or unsafe")
)

# Access the generated schema
schema = criterion.response_schema  # Pydantic model class

# LLM response would instantiate this schema
response = schema(passed=True, passing_criteria="No harmful content detected")
```

### RespField Descriptors

Field descriptors specify what data the LLM should return:

- **TextField**: String text
- **BoolField**: Boolean true/false
- **BoundInt**: Integer with min/max bounds
- **BoundFloat**: Float with min/max bounds
- **ListField**: List of items
- **DictField**: Dictionary with typed values

## Built-in Criteria

### PassFailCriterion

Binary pass/fail evaluation:

```python
from dachi.inst import PassFailCriterion, BoolField, TextField

criterion = PassFailCriterion(
    name="code_quality",
    description="Check if code meets quality standards",
    passed=BoolField(description="Whether code passes quality check"),
    passing_criteria=TextField(description="Explanation of result")
)

# Generated schema expects:
# {
#   "passed": true,
#   "passing_criteria": "Code follows PEP 8, has docstrings, and includes tests"
# }
```

**Fields:**
- `passed`: BoolField - Whether the criterion was met
- `passing_criteria`: TextField - Explanation of criteria

### LikertCriterion

Likert scale (1-5) rating:

```python
from dachi.inst import LikertCriterion, BoundInt

criterion = LikertCriterion(
    name="response_quality",
    description="Rate the quality of the response",
    rating=BoundInt(min_val=1, max_val=5, description="Quality rating from 1-5")
)

# Generated schema expects:
# {
#   "rating": 4
# }
```

**Fields:**
- `rating`: BoundInt(1, 5) - Likert scale rating

### NumericalRatingCriterion

Continuous numerical score:

```python
from dachi.inst import NumericalRatingCriterion, BoundFloat

criterion = NumericalRatingCriterion(
    name="essay_score",
    description="Score essay on scale of 0-10",
    score=BoundFloat(min_val=0.0, max_val=10.0, description="Essay score")
)

# Generated schema expects:
# {
#   "score": 8.5
# }
```

**Fields:**
- `score`: BoundFloat(0.0, 10.0) - Numerical score

### ChecklistCriterion

Multiple boolean checks:

```python
from dachi.inst import ChecklistCriterion, DictField, ListField

criterion = ChecklistCriterion(
    name="requirements_check",
    description="Verify all requirements are met",
    items=DictField(
        value_type=bool,
        description="Checklist items with pass/fail status"
    ),
    missing_items=ListField(
        item_type=str,
        description="List of missing requirements"
    )
)

# Generated schema expects:
# {
#   "items": {
#     "has_docstring": true,
#     "has_tests": false,
#     "follows_style": true
#   },
#   "missing_items": ["tests"]
# }
```

**Fields:**
- `items`: Dict[str, bool] - Item name to pass/fail status
- `missing_items`: List[str] - Items that failed

### HolisticRubricCriterion

Single overall level assessment:

```python
from dachi.inst import HolisticRubricCriterion, TextField, BoundInt

criterion = HolisticRubricCriterion(
    name="writing_quality",
    description="Overall writing quality level",
    level=TextField(description="The achieved level name"),
    level_index=BoundInt(min_val=1, max_val=5, description="Level index")
)

# Generated schema expects:
# {
#   "level": "Proficient",
#   "level_index": 4
# }
```

**Fields:**
- `level`: TextField - Level name (e.g., "Novice", "Proficient")
- `level_index`: BoundInt(1, 5) - Numeric level

### AnalyticRubricCriterion

Multiple dimensions with scores:

```python
from dachi.inst import AnalyticRubricCriterion, DictField, BoundFloat

criterion = AnalyticRubricCriterion(
    name="essay_rubric",
    description="Multi-dimensional essay evaluation",
    dimensions=DictField(
        value_type=dict,
        description="Dimensions with scores and explanations"
    ),
    overall_score=BoundFloat(
        min_val=0.0, max_val=10.0,
        description="Overall score"
    )
)

# Generated schema expects:
# {
#   "dimensions": {
#     "clarity": {"score": 8.5, "explanation": "Very clear structure"},
#     "grammar": {"score": 9.0, "explanation": "Excellent grammar"},
#     "argumentation": {"score": 7.0, "explanation": "Good but could be stronger"}
#   },
#   "overall_score": 8.2
# }
```

**Fields:**
- `dimensions`: Dict[str, dict] - Dimension name to {score, explanation}
- `overall_score`: BoundFloat(0.0, 10.0) - Overall score

### NarrativeCriterion

Qualitative narrative feedback:

```python
from dachi.inst import NarrativeCriterion, TextField

criterion = NarrativeCriterion(
    name="code_review",
    description="Provide detailed code review feedback",
    narrative=TextField(description="Detailed narrative feedback")
)

# Generated schema expects:
# {
#   "narrative": "The code is well-structured and follows best practices..."
# }
```

**Fields:**
- `narrative`: TextField - Qualitative feedback text

### ComparativeCriterion

Ranking or comparison between options:

```python
from dachi.inst import ComparativeCriterion, TextField

criterion = ComparativeCriterion(
    name="response_ranking",
    description="Compare multiple responses",
    mode=TextField(description="Comparison mode: pairwise, ranking, or best_of"),
    result=TextField(description="Comparison result")
)

# Generated schema expects:
# {
#   "mode": "ranking",
#   "result": "response_2, response_1, response_3"
# }
```

**Fields:**
- `mode`: TextField - Comparison type
- `result`: TextField - Winner ID or ranking

## Field Types Reference

### TextField

Free-form text:

```python
from dachi.inst import TextField

field = TextField(description="Explanation text")
# Generated type: str
```

### BoolField

Boolean true/false:

```python
from dachi.inst import BoolField

field = BoolField(description="Pass or fail")
# Generated type: bool
```

### BoundInt

Integer with constraints:

```python
from dachi.inst import BoundInt

field = BoundInt(
    min_val=1,
    max_val=10,
    description="Rating from 1 to 10"
)
# Generated type: int with ge=1, le=10
```

### BoundFloat

Float with constraints:

```python
from dachi.inst import BoundFloat

field = BoundFloat(
    min_val=0.0,
    max_val=1.0,
    description="Probability between 0 and 1"
)
# Generated type: float with ge=0.0, le=1.0
```

### ListField

List of items:

```python
from dachi.inst import ListField

field = ListField(
    item_type=str,
    min_len=1,
    max_len=10,
    description="List of issues found"
)
# Generated type: List[str] with min_items=1, max_items=10
```

### DictField

Dictionary with typed values:

```python
from dachi.inst import DictField

field = DictField(
    value_type=bool,
    description="Feature flags"
)
# Generated type: Dict[str, bool]
```

## Creating Custom Criteria

Extend ResponseSpec to create domain-specific criteria:

```python
from dachi.inst import ResponseSpec, TextField, BoundFloat, ListField

class CodeReviewCriterion(ResponseSpec):
    """Comprehensive code review evaluation."""

    summary: TextField = TextField(
        description="Overall summary of the code review"
    )
    code_quality: BoundFloat = BoundFloat(
        min_val=0.0, max_val=10.0,
        description="Code quality score"
    )
    issues_found: ListField = ListField(
        item_type=str,
        description="List of issues identified"
    )
    recommendations: ListField = ListField(
        item_type=str,
        description="List of recommendations"
    )

# Usage
criterion = CodeReviewCriterion(
    name="thorough_code_review",
    description="Detailed code quality assessment",
    summary=TextField(description="Review summary"),
    code_quality=BoundFloat(0.0, 10.0, description="Quality score"),
    issues_found=ListField(item_type=str, description="Issues"),
    recommendations=ListField(item_type=str, description="Recommendations")
)

# Access generated schema
schema = criterion.response_schema

# Example LLM response
response = schema(
    summary="Good structure but needs more tests",
    code_quality=7.5,
    issues_found=["Missing unit tests", "No error handling in main()"],
    recommendations=["Add pytest tests", "Implement try/except blocks"]
)
```

## Using Criteria with LangCritic

Criteria integrate with LangCritic for LLM-based evaluation:

```python
from dachi.proc import LangCritic
from dachi.inst import PassFailCriterion, BoolField, TextField

# Define criterion
criterion = PassFailCriterion(
    name="safety",
    description="Check content safety",
    passed=BoolField(description="Whether content is safe"),
    passing_criteria=TextField(description="Safety assessment")
)

# Create critic
critic = LangCritic(
    llm=my_llm_model,
    criterion=criterion
)

# Evaluate content
input_text = "Some content to evaluate"
context = {"guidelines": "No harmful content"}

evaluation = await critic.aforward(
    input=input_text,
    context=context
)

# Access results
if evaluation.passed:
    print(f"Safe: {evaluation.passing_criteria}")
else:
    print(f"Unsafe: {evaluation.passing_criteria}")
```

## Batch Evaluation

Criteria automatically generate batch evaluation schemas:

```python
criterion = LikertCriterion(
    name="quality",
    rating=BoundInt(min_val=1, max_val=5)
)

# Single evaluation schema
single_schema = criterion.response_schema

# Batch evaluation schema (automatically created)
batch_schema = criterion.batch_response_schema

# Batch response structure:
# {
#   "responses": [
#     {"rating": 5},
#     {"rating": 3},
#     {"rating": 4}
#   ]
# }
```

## Practical Examples

### Example 1: Code Safety Checker

```python
from dachi.inst import PassFailCriterion, BoolField, TextField, ListField

class CodeSafetyCriterion(ResponseSpec):
    """Evaluate code for security issues."""

    is_safe: BoolField = BoolField(description="Whether code is safe")
    vulnerabilities: ListField = ListField(
        item_type=str,
        description="Security vulnerabilities found"
    )
    severity: TextField = TextField(
        description="Overall severity: low, medium, high, critical"
    )
    recommendation: TextField = TextField(
        description="Recommended actions"
    )

criterion = CodeSafetyCriterion(
    name="security_scan",
    description="Scan code for security vulnerabilities",
    is_safe=BoolField(description="Is the code safe?"),
    vulnerabilities=ListField(item_type=str, description="Vulnerabilities"),
    severity=TextField(description="Severity level"),
    recommendation=TextField(description="What to do")
)
```

### Example 2: Essay Grading

```python
from dachi.inst import AnalyticRubricCriterion, DictField, BoundFloat

criterion = AnalyticRubricCriterion(
    name="essay_grading",
    description="Grade essay across multiple dimensions",
    dimensions=DictField(
        value_type=dict,
        description="Grading dimensions: thesis, evidence, organization, style"
    ),
    overall_score=BoundFloat(
        min_val=0.0, max_val=100.0,
        description="Final percentage score"
    )
)

# LLM returns:
# {
#   "dimensions": {
#     "thesis": {"score": 18, "explanation": "Clear, well-defined thesis"},
#     "evidence": {"score": 22, "explanation": "Strong supporting evidence"},
#     "organization": {"score": 19, "explanation": "Logical flow"},
#     "style": {"score": 17, "explanation": "Engaging writing style"}
#   },
#   "overall_score": 85.0
# }
```

### Example 3: Multi-Criteria Evaluation

Combine multiple criteria for comprehensive evaluation:

```python
from dachi.inst import (
    PassFailCriterion, LikertCriterion, NarrativeCriterion,
    BoolField, BoundInt, TextField
)

# Define multiple criteria
safety = PassFailCriterion(
    name="safety",
    passed=BoolField(description="Content is safe"),
    passing_criteria=TextField(description="Safety explanation")
)

quality = LikertCriterion(
    name="quality",
    rating=BoundInt(min_val=1, max_val=5, description="Quality rating")
)

feedback = NarrativeCriterion(
    name="feedback",
    narrative=TextField(description="Detailed feedback")
)

# Use all three in evaluation workflow
criteria = [safety, quality, feedback]

for criterion in criteria:
    # Evaluate with each criterion
    result = await critic.aforward(
        input=content,
        criterion=criterion
    )
```

## Best Practices

1. **Be Specific**: Write clear field descriptions that guide the LLM

2. **Use Appropriate Bounds**: Set realistic min/max values for numeric fields

3. **Combine Criteria**: Use multiple criteria for comprehensive evaluation

4. **Test Schemas**: Validate that generated schemas match your needs

5. **Name Descriptively**: Use clear, meaningful names for criteria and fields

6. **Provide Context**: Include enough context in descriptions for accurate evaluation

## Integration with Optimization

Criteria enable structured feedback for LangOptim:

```python
from dachi.proc import LangOptim, LangCritic
from dachi.inst import PassFailCriterion

# Define what "good" means
criterion = PassFailCriterion(
    name="prompt_effectiveness",
    passed=BoolField(description="Whether prompt produces desired output"),
    passing_criteria=TextField(description="What makes it effective/ineffective")
)

# Use in optimization loop
critic = LangCritic(llm=llm, criterion=criterion)
optimizer = LangOptim(
    llm=llm,
    params=module.param_set(),
    critic=critic
)

# Optimize parameters based on structured feedback
optimizer.step()
```

## Next Steps

- **[Optimization Guide](optimization-guide.md)** - Use criteria with LangOptim
- **[Process Framework](process-framework.md)** - Integrate LangCritic as a process
- **[LangModel Adapters](langmodel-adapters.md)** - Connect LLMs for evaluation

---

The Criterion system provides structured, type-safe evaluation schemas that enable consistent LLM-based assessment and automated optimization workflows.
