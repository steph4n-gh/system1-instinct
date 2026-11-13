# Offline Prequential Cassette Benchmark

This directory provides the `system1.cassette.v1` JSONL record/replay harness for strictly causal test-then-train evaluation with zero live network access.

## Design

- **Socket-blocked replay.** [cassette.py](cassette.py) monkeypatches `socket.socket.connect` and `socket.create_connection` to raise `NetworkBlockedError` before stream evaluation begins.
- **No label leakage at runtime.** `CassetteTeacher` strips `eval_audit` on load and serves only recorded `response.parsed` payloads indexed by the canonical request SHA-256 key (`provider`, `model`, redacted `prompt_text`, `schema_sha256`, `temperature`, `seed`).
- **Secret scrubbing.** Auth bearer tokens and API key patterns are redacted before hashing or writing cassette fixtures.
- **Canonical fixture.** [support_stream_v1.cassette.jsonl](fixtures/support_stream_v1.cassette.jsonl) exercises warm-start compilation, in-scope local fast-path acceptance, out-of-scope escalation, and repeated queries. Running the 32-step evaluation tail on the causal baseline measures `in_scope_coverage=0.5926` (`16/27`), `conditional_error=0.0`, `oos_false_accept=1/5` (`0.20`), `teacher_calls=15/32`, and `teacher_reduction=0.5312`.

## Usage

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmarks/prequential/run_causal_stream.py --json
```
