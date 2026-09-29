"""A/B a llama.cpp decode command under the power meter.

    python examples/ab_decode.py --a "cmd for baseline" --b "cmd for candidate"

Each command runs while npu-watt records; tokens and tok/s come from llama.cpp's `eval time` line
in the command's output. Prints per-arm reports and the B-vs-A comparison. Use --order ABBA to
cancel drift (thermal state, battery level).
"""
import argparse
import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import npu_watt


def run_arm(label, cmd):
    with npu_watt.measure(label, idle_secs=3) as m:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        for line in (p.stdout + p.stderr).splitlines():
            if "eval time" in line and "prompt eval" not in line:
                m.set_eval_line(line)
    print(m.text)
    return m.report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--order", default="AB", choices=["AB", "ABBA"])
    args = ap.parse_args()
    cmds = {"A": args.a, "B": args.b}
    reports = {"A": [], "B": []}
    for arm in args.order:
        reports[arm].append(run_arm(f"{arm}{len(reports[arm]) + 1}", cmds[arm]))
    # compare first A against last B; with ABBA the middle runs show run-to-run spread
    c = npu_watt.compare(reports["A"][0], reports["B"][-1])
    print(c["summary"])
    if args.order == "ABBA":
        print("A repeat:", npu_watt.compare(reports["A"][0], reports["A"][1])["summary"])
        print("B repeat:", npu_watt.compare(reports["B"][0], reports["B"][1])["summary"])


if __name__ == "__main__":
    main()
