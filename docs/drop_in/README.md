# Replace framework routers and LLM wrappers with Instinct

Each guide below shows how to replace an existing router, classifier, or LLM wrapper (`semantic-router`, `langchain-core`, `FastAPI` with `instructor`, or `poke-env` with `smogon/pokemon-showdown`) with **System 1: Instinct**. The runnable scripts compare latency, accuracy, and out-of-scope rejection against neural encoders (`BAAI/bge-small-en-v1.5` and `sentence-transformers/all-MiniLM-L6-v2`) on public datasets (`CLINC150-OOS`, `BANKING77`, and `NotInject`):

| # | Guide | Target library | Dataset or environment | Runnable script | Measured result |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | [Replace Semantic Router](01_semantic_router.md) | `semantic-router` (`0.1.16`) with `FastEmbedEncoder` (`bge-small-en-v1.5`, `67 MB`) | `CLINC150-OOS` (`24` intents, `144` in-scope and `120` OOS queries) | [`semantic_router_dropin.py`](../../examples/drop_in/semantic_router_dropin.py) | **`0.14 ms` (`30x` faster)**, **`0.0% to 0.8%` OOS false accept** (vs. `100.0%` on `semantic-router`), and **`99.3%` with online correction** |
| **2** | [Replace RouteLLM in LangChain](02_routellm_cascade.md) | `langchain-core` (`RunnableLambda` LCEL) with `all-MiniLM-L6-v2` (`91 MB` ONNX) | `BANKING77` (`32` intents, `256` held-out clean and character-typo queries) | [`routellm_dropin.py`](../../examples/drop_in/routellm_dropin.py) | **`0.22 ms` (`6x` faster)**, **`74.2%` under typo stress** (`+3.5%` vs. `all-MiniLM`), and **`91.4%` with online correction** |
| **3** | [Replace Instructor in FastAPI](03_instructor_structured_outputs.md) | `FastAPI` (`POST /v1/triage`) with `instructor` and Pydantic `BaseModel` | Three-wave `35`-ticket incident stream (four-field schema) | [`instructor_structured_dropin.py`](../../examples/drop_in/instructor_structured_dropin.py) | **Wave 1: `62.9%` local. Wave 2 and Wave 3: `100.0%` local (`0` LLM calls, `100.0%` four-field exact match at `0.455 ms`)** |
| **4** | [Replace Llama-Prompt-Guard](04_prompt_guard_security.md) | `all-MiniLM-L6-v2` (`91 MB` ONNX) and classical linear baseline | `NotInject` benign hard negatives and negation polarity contrasts | [`prompt_guard_dropin.py`](../../examples/drop_in/prompt_guard_dropin.py) | **`0.21 ms` P50**, **`0.0%` `NotInject` benign FPR**, and **`90.0%` static to `100.0%` online accuracy** on negation contrasts |
| **5** | [Wrap an LLM call for `.s1m` cutover](05_llm_observe_and_cutover.md) | `BoundedReviewFallback` wrapping an OpenAI, Anthropic, or Gemini teacher | Three-wave customer ticket stream (`1` teacher call to local cutover) | [`llm_cutover_dropin.py`](../../examples/drop_in/llm_cutover_dropin.py) | **Wave 1: `1` LLM call. Wave 2 and Wave 3: `0` LLM calls (`100%` local at `0.89 ms`)** |
| **6** | [Build a `poke-env` Showdown agent](06_poke_env_showdown_agent.md) | `poke-env` (`0.11.0`) with a local `smogon/pokemon-showdown` WebSocket server | Live `gen9randombattle` against `RandomPlayer`, `MaxBasePowerPlayer`, and `SimpleHeuristicsPlayer` | [`evaluate_poke_env_showdown.py`](../../benchmarks/quality/evaluate_poke_env_showdown.py) | **`100%` vs. `Random`**, **`98%` vs. `MaxBasePower`**, and **`42%` cold to `62%` online vs. `SimpleHeuristics` at `0.093 ms`** |

---

## Run the public dataset benchmarks

To reproduce all adaptation curves and neural baseline comparisons from the repository root, run:

```bash
python benchmarks/quality/evaluate_drop_in_replacements.py
```

For full tables across zero-shot, `adapt_zero_shot`, few-shot, and compiled `.s1m` models, see [drop_in_and_adaptation_report.md](../../benchmarks/quality/drop_in_and_adaptation_report.md).
