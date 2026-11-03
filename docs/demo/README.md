# Recorded Odyssey demo

Recorded on 2 November 2026 from source commit
`c0137fa374cf02b99d5528056e0fc6cb64fab54c`, with the actual Odyssey CLI and
local Ollama model `qwen3.8:27b-mlx`.

The user asks the agent to create a small `greet.py` module and invoke a Python
assertion through the bash tool. `tool-trace.jsonl` preserves the real model
calls and tool results; the bash result is `PASS: greeting verified`. The
recording also includes a separate shell assertion as a cross-check.

- `session.cast`: original ANSI terminal output with original timing, 110×32.
- `tool-trace.jsonl`: saved conversation including file-write and bash results.
- `odyssey-demo.gif` / `.mp4`: replay rendered from the recorded terminal bytes.
  Idle gaps are capped and viewing holds added. No model response or tool output
  is fabricated. This is a functional demo, not a throughput benchmark.
- `preview.png`: a still from the recorded agent response for project previews.
- `capture-summary.json` and `render-manifest.json`: capture and render metadata.

A small Python function is intentionally used to make the tool loop easy to
inspect. This demonstration does not validate arbitrary coding tasks or every
one of Odyssey's tools. The recording uses disposable configuration, data and
workspace directories rather than an existing personal session.

## Capture and rendering

Install the project (`uv sync`) and start Ollama with the named model available.
`python docs/demo/record_session.py` runs the CLI in `demo-workspace/`, with
separate `demo-config/` and `demo-data/` directories. Install Pillow and pyte to
replay this captured session with `python docs/demo/render_session.py`. The
renderer uses the macOS Menlo and Arial fonts and frame selection for this
recording. Read the script before recording a different task.

## Crash-resume demo

`crash_resume.py` is a second, separate demo of the graph engine: it kills the first process with SIGKILL
mid-turn and finishes the turn in a new process from the SQLite checkpoint, then the ledger gate runs pytest.
`crash-resume.txt` is its captured output, `render_crash_resume.py` (needs Pillow) draws that text into
`crash-resume.gif`. The model is a scripted stand-in with three fixed replies; unlike the recording above, it
is not a model result. The captured text is drawn line by line, with timing and colour added and no lines added.
