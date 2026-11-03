# Evaluation report: qwen3.8:27b-mlx

- Date: 2026-11-03 00:26 UTC
- Device: Darwin arm64, Python 3.13.12
- Tools offered to the model: read, write, edit, glob, grep, todo_write
- Suite: version 1, hash `2419f570a45a`
- Packages: odyssey 0.1.0, langgraph 1.2.2, ollama 0.6.2, trajectory-replay 0.1.0, transactional-agent-ledger 0.1.0

| Task | Passed | Gate | Repairs | Model calls | Tool calls | Seconds | Trajectory F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| off_by_one | yes | verified | 0 | 3 | 3 | 33.34 | 0.00 |
| fizzbuzz | yes | verified | 0 | 7 | 9 | 48.24 | 0.00 |
| slugify | yes | verified | 0 | 5 | 6 | 41.45 | 0.00 |
| mutable_default | yes | verified | 0 | 8 | 10 | 64.7 | 0.00 |
| rename_function | yes | verified | 0 | 11 | 15 | 64.55 | 0.00 |
| parse_kv | yes | verified | 0 | 11 | 11 | 78.6 | 0.00 |
| stack_max | yes | verified | 0 | 9 | 12 | 49.65 | 0.00 |
| csv_total | yes | verified | 0 | 4 | 5 | 25.64 | 0.00 |
| validate_person | yes | verified | 0 | 3 | 3 | 45.55 | 0.00 |
| top_scores | yes | verified | 0 | 4 | 5 | 33.63 | 0.00 |

**Passed 10 of 10** (independent check, test files pinned). The gate reported verified for 10; it disagreed with the independent check on 0.

Trajectory F1 compares the model's tool calls with the reference solution's. A coding task has many valid trajectories (reading first, editing instead of writing), so treat it as a diagnostic, not a score.
