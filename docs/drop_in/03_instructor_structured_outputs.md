# Drop-In Guide 3: Replace `Instructor` / `BAML` (Structured Extraction)

Replace `instructor.from_openai(client).chat.completions.create(response_model=...)` with [`SystemOneInstructorClient`](../../examples/drop_in/instructor_structured_dropin.py) whenever your schema consists of **bounded classifications, boolean flags, multi-label tags, and numerical scores**.

Instead of waiting **500–2,000 ms** for an LLM to generate JSON token-by-token, System 1 evaluates all schema fields in parallel in **~2 ms** with zero syntax errors.

---

## 1. Before vs. After

### Before (`instructor` + OpenAI API)
```python
import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal, List

class IncidentTriage(BaseModel):
    department: Literal["security_ops", "billing_finance", "platform_infra", "customer_success"]
    page_oncall: bool
    impact_tags: List[Literal["auth", "payments", "database", "api"]]
    urgency_score: float

client = instructor.from_openai(OpenAI())
# 800+ ms network call + token generation cost on every ticket
result = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=IncidentTriage,
    messages=[{"role": "user", "content": "Critical production outage: postgres database deadlock causing 502 gateway API timeouts"}],
)
```

### After (`System 1: Instinct` — ~2 ms local, 0 LLM calls)
```python
from examples.drop_in import IncidentTriageModel, SystemOneInstructorClient

client = SystemOneInstructorClient(response_model=IncidentTriageModel)
result = client.create(
    "Critical production outage: postgres database deadlock causing 502 gateway API timeouts"
)
print(result.values)
# -> {'department': 'platform_infra', 'page_oncall': True, 'impact_tags': ('database', 'api'), 'urgency_score': 0.70}
```

---

## 2. Step-by-Step Procedure

### Step 1: Declare Your Multi-Field `DecisionSchema`
Map each Pydantic field to its System 1 decision head in [`IncidentTriageModel`](../../examples/drop_in/instructor_structured_dropin.py):
- `ChoiceField` $\to$ `Literal[...]` / `Enum`
- `BooleanField` $\to$ `bool`
- `MultiChoiceField` $\to$ `List[Literal[...]]`
- `ScoreField` $\to$ bounded `float`

### Step 2: Extract All Fields in One Pass
Call `client.create(prompt)`. All four heads share a single text projection pass and return a typed `StructuredExtractionResult`.

### Step 3: Correct Any Field Online in 1 Shot (`teach_example`)
If a new ticket pattern needs a correction across all 4 fields simultaneously:

```python
client.teach_example(
    "BGP route leak in us-east-1 causing packet drop across edge routers",
    {
        "department": "platform_infra",
        "page_oncall": True,
        "impact_tags": ["api"],
        "urgency_score": 0.95,
    },
)
```

---

## 3. Run It & See the Real Result

Run the standalone script from the repository root:

```bash
python examples/drop_in/instructor_structured_dropin.py
```

### Real Output
```text
=== System 1: Instinct — Instructor Structured Extraction Drop-In Demo ===
  dept=security_ops       | page_oncall=True  | tags=('auth',)                | urgency=0.68 | latency=2.091 ms
  dept=billing_finance    | page_oncall=False | tags=('payments',)            | urgency=0.46 | latency=2.231 ms
  dept=platform_infra     | page_oncall=True  | tags=('database', 'api')      | urgency=0.70 | latency=1.885 ms
  dept=customer_success   | page_oncall=False | tags=()                       | urgency=0.40 | latency=1.736 ms
```

### Benchmark Scorecard (`evaluate_drop_in_replacements.py`)
- **Department Accuracy**: **`100.0%`**
- **Page On-Call Boolean Accuracy**: **`100.0%`**
- **4-Field Joint Exact Match**: **`100.0%`**
- **Median Latency (P50)**: **`2.729 ms`**
