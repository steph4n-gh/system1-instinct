# System 1: Instinct — Real Framework Drop-In Guides

Every guide below drops **System 1: Instinct** directly into a **real, installed open-source library** (`semantic-router`, `langchain-core`, `FastAPI` + `instructor`, `poke-env` + `smogon/pokemon-showdown`) and runs side-by-side against external neural encoders (`BAAI/bge-small-en-v1.5`, `sentence-transformers/all-MiniLM-L6-v2`) on public datasets (`CLINC150-OOS`, `BANKING77`, `NotInject`):

| # | Drop-In Guide | Real Framework / Stack | Public Dataset / Environment | Runnable Script | Measured Result |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | [Semantic Router Drop-In](01_semantic_router.md) | `semantic-router` (`0.1.16`) + `FastEmbedEncoder` (`bge-small-en-v1.5`, `67 MB`) | `CLINC150-OOS` (`24` intents, `144` in-scope + `120` OOS queries) | [`semantic_router_dropin.py`](../../examples/drop_in/semantic_router_dropin.py) | **`0.14 ms` (`30×` faster)** · **`0.0%–0.8%` OOS False Accept** (vs `100.0%` on `semantic-router`) · **`99.3%` Flywheel** |
| **2** | [LangChain / RouteLLM Drop-In](02_routellm_cascade.md) | `langchain-core` (`RunnableLambda` LCEL) + `all-MiniLM-L6-v2` (`91 MB` ONNX) | `BANKING77` (`32` intents, `256` held-out clean + character-typo queries) | [`routellm_dropin.py`](../../examples/drop_in/routellm_dropin.py) | **`0.22 ms` (`6×` faster)** · **`74.2%` Typo Stress** (`+3.5%` vs `all-MiniLM`) · **`91.4%` Flywheel** |
| **3** | [FastAPI + Instructor Drop-In](03_instructor_structured_outputs.md) | `FastAPI` (`POST /v1/triage`) + `instructor` / Pydantic `BaseModel` | 3-Wave `35`-Ticket Production Incident Stream (4-field schema) | [`instructor_structured_dropin.py`](../../examples/drop_in/instructor_structured_dropin.py) | **Wave 1: `62.9%` local $\to$ Wave 2 & 3: `100.0%` local (`0` LLM calls, `100.0%` 4-field exact match at `0.455 ms`)** |
| **4** | [Prompt-Guard & Polarity Drop-In](04_prompt_guard_security.md) | `all-MiniLM-L6-v2` (`91 MB` ONNX) + Classical Linear Baseline | `NotInject` Benign Hard Negatives + Negation/Polarity Contrasts | [`prompt_guard_dropin.py`](../../examples/drop_in/prompt_guard_dropin.py) | **`0.21 ms` P50** · **`0.0%` `NotInject` Benign FPR** · **`90.0%` static $\to$ `100.0%` flywheel** on negation contrasts |
| **5** | [LLM Observe $\to$ Local `.s1m` Cutover](05_llm_observe_and_cutover.md) | `TypeSafeClient` wrapping any OpenAI / Anthropic / Gemini teacher | 2-Wave Customer Ticket Stream (`6` teacher calls $\to$ auto-compile) | [`llm_cutover_dropin.py`](../../examples/drop_in/llm_cutover_dropin.py) | **Wave 1: `6` LLM calls $\to$ Wave 2: `0` LLM calls (`100%` local at `0.37 ms`)** |
| **6** | [`poke-env` Showdown Reflex Agent](06_poke_env_showdown_agent.md) | `poke-env` (`0.11.0`) + local `smogon/pokemon-showdown` WebSocket server | Live `gen9randombattle` vs `RandomPlayer`, `MaxBasePowerPlayer`, `SimpleHeuristicsPlayer` | [`evaluate_poke_env_showdown.py`](../../benchmarks/quality/evaluate_poke_env_showdown.py) | **`100%` vs `Random`** · **`98%` vs `MaxBasePower`** · **`42%` cold $\to$ `62%` online vs `SimpleHeuristics` at `0.093 ms`** |

---

## Run All Public Dataset Adaptation Curves & Neural Baselines

```bash
python benchmarks/quality/evaluate_drop_in_replacements.py
```

See the full public dataset adaptation curves (`0`-shot $\to$ `adapt_zero_shot` $\to$ `1/3/5`-shot $\to$ `.s1m`) and neural embedding baseline comparison in [drop_in_and_adaptation_report.md](../../benchmarks/quality/drop_in_and_adaptation_report.md).

