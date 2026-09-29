"""npu_watt: use the npu-watt power meter from test and research scripts.

    import npu_watt

    with npu_watt.measure("baseline") as m:
        tok, rate = run_my_decode()          # your workload
        m.tokens, m.tok_s = tok, rate        # or m.set_eval_line(llama_cpp_eval_line)
    base = m.report

    with npu_watt.measure("tile16") as m:
        ...
    print(npu_watt.compare(base, m.report)["summary"])
    # tile16 vs baseline: decode -8.0% (25.00 -> 23.00 tok/s), NPU power -36.0% ...

The meter runs as a `npu-watt record` subprocess. Set NPU_WATT_EXE to the binary if it is not on PATH.
"""
import json
import os
import shutil
import subprocess
import tempfile

__all__ = ["measure", "compare", "average", "find_exe", "Measurement", "NpuVerdictError"]


class NpuVerdictError(RuntimeError):
    """The run did not get the verdict passed as `require=`; `.report` still holds the numbers."""

    def __init__(self, msg, report):
        super().__init__(msg)
        self.report = report


def find_exe():
    exe = os.environ.get("NPU_WATT_EXE") or shutil.which("npu-watt")
    if not exe:
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(here, "..", "target", "release", "npu-watt.exe")
        if os.path.exists(cand):
            exe = cand
    if not exe:
        raise FileNotFoundError("npu-watt not found: cargo install --git https://github.com/hotschmoe/npu-watt, or set NPU_WATT_EXE")
    return exe


class Measurement:
    """Context manager. The idle baseline is measured on entry (idle_secs) before the body runs."""

    def __init__(self, label="run", pids=None, idle_secs=2.0, interval_ms=250, require=None, exe=None):
        self.label, self.tokens, self.tok_s = label, None, None
        self.report = None
        self._eval_lines = []
        args = [exe or find_exe(), "record", "--label", label, "--idle-secs", str(idle_secs), "--interval-ms", str(interval_ms)]
        if require:
            args += ["--require", require]
        for p in pids or []:
            args += ["--pid", str(p)]
        self._args = args
        self._proc = None

    def set_eval_line(self, line):
        """Give a llama.cpp `eval time = ... tokens per second` line; tokens and tok/s are parsed from it."""
        self._eval_lines.append(line)

    def __enter__(self):
        self._proc = subprocess.Popen(self._args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ready = self._proc.stdout.readline().strip()
        if ready != "READY":
            err = self._proc.stderr.read()
            raise RuntimeError(f"npu-watt did not start: {err.strip()}")
        return self

    def __exit__(self, exc_type, exc, tb):
        p = self._proc
        msgs = [f"label={self.label}"]
        if self.tokens is not None:
            msgs.append(f"tokens={self.tokens}")
        if self.tok_s is not None:
            msgs.append(f"tok_s={self.tok_s}")
        msgs += [f"eval={l}" for l in self._eval_lines]
        msgs.append("stop")
        try:
            out, err = p.communicate("\n".join(msgs) + "\n", timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            raise
        self.text = err
        if p.returncode not in (0, 2):
            raise RuntimeError(f"npu-watt failed: {err.strip()}")
        self.report = json.loads(out.strip().splitlines()[-1])
        if p.returncode == 2 and exc_type is None:
            raise NpuVerdictError(err.strip().splitlines()[-1], self.report)
        return False


def measure(label="run", **kw):
    return Measurement(label, **kw)


def compare(a, b, exe=None):
    """B relative to A. Returns a dict with percent changes and a `summary` sentence."""
    with tempfile.TemporaryDirectory() as d:
        pa, pb, pc = (os.path.join(d, n) for n in ("a.json", "b.json", "c.json"))
        for path, rep in ((pa, a), (pb, b)):
            with open(path, "w") as f:
                json.dump(rep, f)
        subprocess.run([exe or find_exe(), "compare", pa, pb, "--json", pc], check=True, capture_output=True, text=True)
        with open(pc) as f:
            return json.load(f)


def average(reports, label=None):
    """Mean of several reports (e.g. the A runs of an ABBA sequence), usable with compare()."""
    def mean(vals):
        if all(isinstance(v, dict) for v in vals):
            keys = set().union(*vals)
            return {k: mean([v[k] for v in vals if k in v]) for k in keys}
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            return sum(vals) / len(vals)
        return vals[0]
    out = mean(reports)
    out["label"] = label or "+".join(r["label"] for r in reports)
    return out
