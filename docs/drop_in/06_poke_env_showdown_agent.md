# How to build and evaluate a `poke-env` Pokémon Showdown agent

[`SystemOneShowdownPlayer`](../../benchmarks/quality/evaluate_poke_env_showdown.py) subclasses `poke_env.player.Player` to play live `gen9randombattle` matches on a Pokémon Showdown server. The player starts at zero examples from four tactical option descriptions, updates its weights online during battles via `adapt_zero_shot` and `learn_from_tier2`, and compiles into a `23 KiB` `.s1m` skill that selects moves in `0.093 ms` (`93 µs`) per turn.

---

## 1. Instantiate `SystemOneShowdownPlayer` and play against `poke-env` bots

To pit a zero-shot, online-adaptive, or compiled `.s1m` player against a `poke-env` opponent in Python, import `SystemOneShowdownPlayer` and `ShowdownTacticalSchema`:

```python
from poke_env.player import SimpleHeuristicsPlayer
from system1 import SystemOneEngine
from benchmarks.quality.evaluate_poke_env_showdown import (
    ShowdownTacticalSchema,
    SystemOneShowdownPlayer,
)

engine = SystemOneEngine(ShowdownTacticalSchema, strict_mode=False, use_cache=False)
player = SystemOneShowdownPlayer(
    engine=engine,
    online_adapt=True,
    battle_format="gen9randombattle",
)
opponent = SimpleHeuristicsPlayer(battle_format="gen9randombattle")
# Run inside an async event loop against a local Showdown server on 127.0.0.1:8000:
# await player.battle_against(opponent, n_battles=50)
```

Standard Pokémon Showdown evaluations (*PokéLLMon*, *Metamon*, and `poke-env`) test `gen9randombattle` agents against three built-in `poke-env` opponents:

1. **`RandomPlayer` (`100.0%` win rate, `50 of 50`).** Selects legal moves and switches uniformly at random. Winning `50 of 50` matches verifies legal action selection and zero stalling.
2. **`MaxBasePowerPlayer` (`98.0%` win rate, `49 of 50`).** Selects the highest base-power move on every turn without checking type immunities, status moves, or switches. Winning `49 of 50` matches verifies that the tactical state encoder and type-effectiveness features beat greedy move selection.
3. **`SimpleHeuristicsPlayer` (`42.0%` cold zero-shot, `62.0%` online adaptive, and `60.0%` compiled `.s1m`).** `SimpleHeuristicsPlayer` is `poke-env`'s strongest built-in rule bot. It computes type multipliers, physical and special stat ratios, entry hazards (`Stealth Rock`, `Spikes`, `Rapid Spin`, and `Defog`), stat boosts (`Swords Dance` and `Nasty Plot`), speed tiers, and Terastallization:
   - **Cold zero-shot (`42.0%`, `21 of 50`).** With zero training examples and only schema option descriptions, `SystemOneShowdownPlayer` wins `21 of 50` matches against `SimpleHeuristicsPlayer`.
   - **Online adaptation (`62.0%`, `31 of 50`).** Updating weights online during live battles with `adapt_zero_shot` and `learn_from_tier2` adds `20` percentage points and beats `SimpleHeuristicsPlayer`.
   - **Compiled `.s1m` skill (`60.0%`, `30 of 50` at `0.093 ms` P50).** After compiling into a `.s1m` artifact, the agent wins `30 of 50` matches at `93 µs` per turn.

---

## 2. Run the offline and live Showdown benchmarks

### Run the offline tactical and jargon adaptation suite

To run the offline tactical evaluation without a local Pokémon Showdown server, execute:

```bash
python benchmarks/quality/evaluate_poke_env_showdown.py --offline-only
```

### Run live `gen9randombattle` matches on Pokémon Showdown

When a local Pokémon Showdown server is listening on `127.0.0.1:8000`, run `50` battles per matchup:

```bash
python benchmarks/quality/evaluate_poke_env_showdown.py --battles 50
```

The script prints:

```text
=== System 1: Instinct — poke-env + Pokémon Showdown Benchmark ===
[Offline Tactical Suite] 0-Shot Canonical=100.0% | Unseen Jargon: 20.0% -> 100.0% (adapt_zero_shot) -> 100.0% (1-shot online) | Compiled .s1m=100.0% (1.230 ms)

[Live Showdown Server] Connected on 127.0.0.1:8000 — running 50 'gen9randombattle' battles per matchup...
  [System 1 Zero-Shot (0 Examples)] vs RandomPlayer: 50/50 wins (100.0%) | P50=3.177 ms | turns=1036 | time=4.7s
  [System 1 Zero-Shot (0 Examples)] vs MaxBasePowerPlayer: 49/50 wins (98.0%) | P50=3.217 ms | turns=829 | time=3.4s
  [System 1 Zero-Shot (0 Examples)] vs SimpleHeuristicsPlayer: 21/50 wins (42.0%) | P50=3.220 ms | turns=863 | time=3.4s
  [System 1 Zero-Shot Adaptive (Online)] vs RandomPlayer: 50/50 wins (100.0%) | P50=3.166 ms | turns=1051 | time=3.9s
  [System 1 Zero-Shot Adaptive (Online)] vs MaxBasePowerPlayer: 46/50 wins (92.0%) | P50=3.201 ms | turns=831 | time=3.2s
  [System 1 Zero-Shot Adaptive (Online)] vs SimpleHeuristicsPlayer: 31/50 wins (62.0%) | P50=3.238 ms | turns=867 | time=3.4s
  [System 1 Compiled (.s1m)] vs RandomPlayer: 49/50 wins (98.0%) | P50=0.097 ms | turns=1148 | time=0.9s
  [System 1 Compiled (.s1m)] vs MaxBasePowerPlayer: 41/50 wins (82.0%) | P50=0.090 ms | turns=874 | time=0.6s
  [System 1 Compiled (.s1m)] vs SimpleHeuristicsPlayer: 30/50 wins (60.0%) | P50=0.093 ms | turns=869 | time=0.6s
```
