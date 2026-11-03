# Evaluation report: scripted reference solution

> **Scripted run. This is not a model result.** It validates the harness: the reference solution must pass every task and a do-nothing model must fail every task.

- Date: 2026-11-03 00:17 UTC
- Device: Darwin arm64, Python 3.13.12
- Tools offered to the model: read, write, edit, glob, grep, todo_write
- Suite: version 1, hash `2419f570a45a`
- Packages: odyssey 0.1.0, langgraph 1.2.2, ollama 0.6.2, trajectory-replay 0.1.0, transactional-agent-ledger 0.1.0

| Task | Passed | Gate | Repairs | Model calls | Tool calls | Seconds | Trajectory F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| off_by_one | yes | verified | 0 | 2 | 1 | 1.75 | 1.00 |
| fizzbuzz | yes | verified | 0 | 2 | 1 | 1.71 | 1.00 |
| slugify | yes | verified | 0 | 2 | 1 | 1.79 | 1.00 |
| mutable_default | yes | verified | 0 | 2 | 1 | 1.74 | 1.00 |
| rename_function | yes | verified | 0 | 3 | 2 | 1.76 | 1.00 |
| parse_kv | yes | verified | 0 | 2 | 1 | 1.75 | 1.00 |
| stack_max | yes | verified | 0 | 2 | 1 | 1.81 | 1.00 |
| csv_total | yes | verified | 0 | 2 | 1 | 1.85 | 1.00 |
| validate_person | yes | verified | 0 | 2 | 1 | 1.81 | 1.00 |
| top_scores | yes | verified | 0 | 2 | 1 | 1.8 | 1.00 |

**Passed 10 of 10** (independent check, test files pinned). The gate reported verified for 10; it disagreed with the independent check on 0.

Trajectory F1 compares the model's tool calls with the reference solution's. A coding task has many valid trajectories (reading first, editing instead of writing), so treat it as a diagnostic, not a score.
