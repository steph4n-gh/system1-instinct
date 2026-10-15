# How to replace `Instructor` inside a `FastAPI` endpoint

Use [`SystemOneInstructorClient`](../../examples/drop_in/instructor_structured_dropin.py) inside a `FastAPI` (`POST /v1/triage`) route to extract four typed fields (`ChoiceField`, `BooleanField`, `MultiChoiceField`, and `ScoreField`) in `0.45 ms` on CPU, escalating to `Instructor` only when the local decision is ambiguous.

---

## 1. Review the existing `FastAPI` and `Instructor` route

In a standard `Instructor` microservice, every `POST /v1/triage` request calls the upstream LLM:

```python
import instructor
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()
client = instructor.from_openai(OpenAI())

class TriageResponse(BaseModel):
    department: str
    page_oncall: bool
    impact_tags: list[str]
    urgency_score: float

@app.post("/v1/triage", response_model=TriageResponse)
def triage_ticket(text: str) -> TriageResponse:
    # Spends 400 to 900 ms and an API call on every HTTP request:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=TriageResponse,
        messages=[{"role": "user", "content": text}],
    )
```

---

## 2. Replace the route handler with `SystemOneInstructorClient`

To serve confident tickets locally in `0.45 ms` and escalate only uncertain tickets (`is_ambiguous=True`) to `Instructor`, call `SystemOneInstructorClient(IncidentTriageModel)` inside your route handler (full code in [`examples/drop_in/instructor_structured_dropin.py`](../../examples/drop_in/instructor_structured_dropin.py)):

```python
from fastapi import FastAPI
from examples.drop_in.instructor_structured_dropin import (
    IncidentTriageModel,
    SystemOneInstructorClient,
    TicketRequest,
    TriageResponse,
)

app = FastAPI()
instinct_client = SystemOneInstructorClient(IncidentTriageModel)

@app.post("/v1/triage", response_model=TriageResponse)
def triage_ticket(req: TicketRequest) -> TriageResponse:
    # Extracts all 4 fields in ~0.45 ms on CPU:
    res = instinct_client.create(req.text)
    return TriageResponse(
        department=res.values["department"],
        page_oncall=res.values["page_oncall"],
        impact_tags=list(res.values["impact_tags"]),
        urgency_score=res.values["urgency_score"],
        latency_ms=res.latency_ms,
    )
```

---

## 3. Run the three-wave `FastAPI` cutover script

To run the three-wave `FastAPI` and `Instructor` cutover across `35` incident tickets per wave, execute:

```bash
python examples/drop_in/instructor_structured_dropin.py
```

### Expected output

```text
=== 1. Single-Pass 4-Field Structured Extraction (`SystemOneInstructorClient`) ===
  dept=security_ops       | page_oncall=True  | tags=('auth',)                | urgency=0.68 | latency=2.078 ms
  dept=billing_finance    | page_oncall=False | tags=('payments',)            | urgency=0.46 | latency=1.845 ms
  dept=platform_infra     | page_oncall=True  | tags=('database', 'api')      | urgency=0.70 | latency=1.806 ms
  dept=customer_success   | page_oncall=False | tags=()                       | urgency=0.40 | latency=2.539 ms

=== 2. Real FastAPI `POST /v1/triage` + Instructor 4-Field Auto-Cutover (35 Tickets x 3 Waves) ===
  [FastAPI Sample HTTP 200] {'department': 'platform_infra', 'page_oncall': True, 'impact_tags': ['database', 'api'], 'urgency_score': 0.696, 'latency_ms': 2.768}
  [Compiled .s1m = 63.8 KiB] Joint 4-Field Exact Match = 100.0% | P50 Latency = 0.455 ms
  Wave 1: Local Instinct Handled= 62.9% | Instructor LLM Escalations=37.1% | End-to-End 4-Field Accuracy=100.0%
  Wave 2: Local Instinct Handled=100.0% | Instructor LLM Escalations= 0.0% | End-to-End 4-Field Accuracy=100.0%
  Wave 3: Local Instinct Handled=100.0% | Instructor LLM Escalations= 0.0% | End-to-End 4-Field Accuracy=100.0%
```
