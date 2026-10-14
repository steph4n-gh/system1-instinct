# System 1: Instinct — Drop-In Replacement Examples

Each script in this directory is a standalone, runnable drop-in replacement that executes **100% locally** with **zero external API keys**.

For step-by-step "Before vs. After" guides and explanations, see **[docs/drop_in/README.md](../../docs/drop_in/README.md)**.

## Quick Run Commands

```bash
# 1. Semantic Router drop-in (0-shot routing + OOD rejection + adapt_zero_shot)
python examples/drop_in/semantic_router_dropin.py

# 2. RouteLLM 3-tier model cascade controller
python examples/drop_in/routellm_dropin.py

# 3. Instructor / BAML 4-field structured output extractor
python examples/drop_in/instructor_structured_dropin.py

# 4. Llama-Prompt-Guard / Guardrails security scanner (0% NotInject benign FPR)
python examples/drop_in/prompt_guard_dropin.py

# 5. Live LLM Observe -> Online RLS -> Local .s1m Cutover
python examples/drop_in/llm_cutover_dropin.py

# Run the complete benchmark + BANKING77/CLINC150 adaptation curve suite
python benchmarks/quality/evaluate_drop_in_replacements.py
```

## Guides & Measured Results

| Guide | Script | What It Replaces | Measured Accuracy | P50 Latency |
| :--- | :--- | :--- | :--- | :---: |
| [Guide 1: Semantic Router](../../docs/drop_in/01_semantic_router.md) | [semantic_router_dropin.py](semantic_router_dropin.py) | `semantic_router.RouteLayer` | **100% In-Domain / 100% OOD / 100% Adapted** | `1.36 ms` |
| [Guide 2: RouteLLM Cascade](../../docs/drop_in/02_routellm_cascade.md) | [routellm_dropin.py](routellm_dropin.py) | `routellm.controller.Controller` | **100% Tier Routing** | `1.47 ms` |
| [Guide 3: Instructor / BAML](../../docs/drop_in/03_instructor_structured_outputs.md) | [instructor_structured_dropin.py](instructor_structured_dropin.py) | `instructor.from_openai(...).create()` | **100% 4-Field Joint Match** | `2.73 ms` |
| [Guide 4: Prompt-Guard](../../docs/drop_in/04_prompt_guard_security.md) | [prompt_guard_dropin.py](prompt_guard_dropin.py) | `Llama-Prompt-Guard` / `Guardrails` | **0.0% Benign FPR / 100% Attack Recall** | `0.81 ms` |
| [Guide 5: LLM Auto-Cutover](../../docs/drop_in/05_llm_observe_and_cutover.md) | [llm_cutover_dropin.py](llm_cutover_dropin.py) | Recurring OpenAI / Anthropic classification | **100% Local Cutover by Wave 2** | `0.37 ms` |
| [Guide 6: `poke-env` Showdown](../../docs/drop_in/06_poke_env_showdown_agent.md) | [evaluate_poke_env_showdown.py](../../benchmarks/quality/evaluate_poke_env_showdown.py) | Slow LLM / static game bots | **62% vs `SimpleHeuristicsPlayer`** | `0.093 ms` |
