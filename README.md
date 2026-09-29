# npu-watt

Per-rail power and energy-per-token measurement for NPU-pinned LLM inference on Snapdragon X
(Windows on ARM). MIT licensed.

**Why:** Task Manager's NPU graph can read 0% while the Hexagon cDSP is saturated, because work that
reaches the DSP over FastRPC (llama.cpp's HTP backend, QNN, ...) is not a WDDM engine. There is no
utilization counter for it. The Windows Energy Meter does expose cumulative energy per rail
(`npu`, `gpu`, `cpu_cluster_*`, `memory`, `soc`, `system`), which is enough to answer:

- Is the NPU actually doing the work, and is the CPU/GPU staying out of it?
- How many joules per token, and tokens/s per watt? (Chasing efficiency, not just speed.)
- How long does a battery last doing this?

Example (Qwen3.6-35B-A3B Q4_0 on the Hexagon NPU, 250 tokens, on battery):
npu +3.6 W, gpu +0.01 W, process tree CPU well under a core, 0.150 J/token, 6.7 tok/s/W, verdict `npu-only`.

## Install

```
cargo install --git https://github.com/hotschmoe/npu-watt
```

Needs a Windows machine with an Energy Meter (Snapdragon X laptops have one; check with
`npu-watt monitor`). Build target `aarch64-pc-windows-msvc`; x64 Windows works if the meter exists.

## CLI

```
npu-watt monitor [--interval-ms N]
npu-watt run [--label L] [--json F] [--csv F] [--tokens N] [--tok-s X] [--idle-secs S] -- <cmd> [args]
npu-watt record [--label L] [--idle-secs S] [--pid P]...        # for scripts, see below
npu-watt compare A.json B.json [--json OUT.json]
npu-watt show REPORT.json...                                    # re-render saved reports
```

`run` and `record` take `--require npu-only` and exit with code 2 if the verdict differs, so a test can
fail loudly if a change makes the GPU or CPU pick up work. In Python, `measure(..., require="npu-only")`
raises `NpuVerdictError` (its `.report` still holds the numbers).

`run` measures an idle baseline, runs your command in a job object (so CPU time of the whole process
tree counts), and reports over the **NPU-active window** (intervals where npu draw is clearly above
idle): per-rail watts, delta vs idle, joules, a verdict, J/token, tok/s/W, and battery figures.
Tokens and tok/s are parsed from llama.cpp's `eval time = ... tokens per second` line, or pass
`--tokens/--tok-s`.

Verdicts: `npu-only` (npu above idle, GPU within 0.3 W of idle, process tree under 0.35 cores),
`npu-mixed` (NPU active but GPU or CPU also busy), `npu-inactive`.

## Use it from test and research scripts

Python (`python/npu_watt.py`, no dependencies; it drives `npu-watt record`):

```python
import npu_watt

with npu_watt.measure("baseline") as m:
    tokens, rate = run_decode()          # your workload
    m.tokens, m.tok_s = tokens, rate     # or m.set_eval_line(llama_cpp_eval_line)
base = m.report

with npu_watt.measure("tile16") as m:
    ...
print(npu_watt.compare(base, m.report)["summary"])
# tile16 vs baseline: decode -8.0% (25.00 -> 23.00 tok/s), NPU power -36.0% (4.00 -> 2.56 W), NPU J/token -30.4%, ...
```

`npu_watt.average(reports)` averages repeated runs (the A runs of an ABBA sequence) so `compare` works on
means. `examples/ab_decode.py --a "<cmd>" --b "<cmd>" [--order ABBA]` A/Bs two llama.cpp commands.

Rust: the crate is also a library (`Meter`, `Recorder`, `analyze`, `compare`, `Report`); see the docs in
`src/lib.rs`. `record` speaks a tiny protocol if you want another language: it prints `READY` after
the idle baseline, records until stdin closes or a `stop` line (optional `tokens=N`, `tok_s=X`,
`label=L`, `eval=<llama.cpp line>` lines before it), then prints the JSON report as the last stdout line.

## Practical notes

- **Compare J/token, not just tok/s.** Identical runs varied 4.7% in decode speed but 0.1% in NPU
  J/token, so energy per token is the steadier A/B metric.
- `system` includes the display and platform overhead. The battery-reported draw is coarse over short
  windows; unplug and use longer runs (thousands of tokens) for battery-life numbers.
- The idle baseline is a short sample. Use `--idle-secs 5` on a quiet machine for cleaner "vs idle".
- The Energy counter is cumulative picowatt-hours; energy is integrated from it, not from sampled power.
- Alternate A and B (ABBA) to cancel thermal and battery-level drift.

Build and test: `cargo test`, `cargo build --release`.
