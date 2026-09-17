"""
Live progress view for a running full sweep.

    python experiments/full_sweep/watch_progress.py            # live, redraws
    python experiments/full_sweep/watch_progress.py --once     # one snapshot
    python experiments/full_sweep/watch_progress.py --out_dir results/x

Reads ``progress.jsonl`` (one fsync'd line per finished cell) and
``run_config.json``, so it is safe to start, stop and restart at any time --
it never touches the sweep itself.  Ctrl-C exits the viewer, not the sweep.
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

THETA_STARS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PSI_TRUES = ("inf", "pers+", "pers-")
N_ALPHAS = 11


def _fmt_dur(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def _read_records(out_dir: str):
    path = os.path.join(out_dir, "progress.jsonl")
    if not os.path.isfile(path):
        return []
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a line still being appended; skip it
    return recs


def _read_config(out_dir: str):
    path = os.path.join(out_dir, "run_config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _disk_bytes(out_dir: str) -> int:
    root = os.path.join(out_dir, "trajectories")
    total = 0
    if not os.path.isdir(root):
        return 0
    for name in os.listdir(root):
        p = os.path.join(root, name, "part.parquet")
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
    return total


def _elapsed(out_dir: str, cfg: dict, status: str) -> float:
    """
    Seconds since the run started.

    ``run_config.json`` is written at start-up with ``status: running``, so
    while the sweep is live its ``run_date_utc`` *is* the start time.  Once
    the run ends the file is rewritten, and that timestamp becomes the end
    time -- so a finished run reports its recorded ``wall_seconds`` instead.

    Do not measure from ``progress.jsonl``: that file is only created when
    the *first cell finishes*, which understates elapsed by a full cell and
    badly inflates the early rate.
    """
    if status != "running":
        return float(cfg.get("wall_seconds") or 0.0)
    stamp = cfg.get("run_date_utc")
    if stamp:
        try:
            return max(0.0, time.time()
                       - calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")))
        except (ValueError, OverflowError):
            pass
    try:                                    # last resort: journal age
        return time.time() - os.path.getctime(
            os.path.join(out_dir, "progress.jsonl"))
    except OSError:
        return 0.0


def _free_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return None


def render(out_dir: str) -> tuple[str, bool]:
    """Return (screen, finished)."""
    cfg = _read_config(out_dir)
    recs = _read_records(out_dir)
    grid = cfg.get("grid", {})
    n_cells = grid.get("n_cells", len(THETA_STARS) * len(PSI_TRUES) * N_ALPHAS)
    status = cfg.get("status", "unknown")

    done = len(recs)
    rows = sum(r.get("rows", 0) for r in recs)
    frac = (done / n_cells) if n_cells else 0.0

    L = []
    L.append(f"full sweep  ->  {out_dir}")
    L.append("")

    width = 44
    filled = int(width * frac)
    bar = "#" * filled + "." * (width - filled)
    L.append(f"  [{bar}] {done}/{n_cells} cells ({100*frac:5.1f}%)")
    L.append("")

    if recs:
        cell_secs = [r.get("wall_s", 0.0) for r in recs]
        mean_cell = sum(cell_secs) / len(cell_secs)
        elapsed = _elapsed(out_dir, cfg, status)
        rate_min = (done / elapsed * 60) if elapsed > 0 else 0.0
        remaining = n_cells - done

        # Cells finish in near-simultaneous waves of n_workers, so dividing by
        # wall-clock *now* charges the not-yet-landed wave against the rate and
        # inflates the ETA. Measure throughput up to the last completion
        # instead -- progress.jsonl's mtime is exactly that instant -- and
        # recover effective concurrency from the cell durations.
        eta = float("inf")
        if remaining and status == "running":     # a stopped run has no ETA
            workers = cfg.get("workers")
            if workers:
                eta = remaining * mean_cell / workers
            else:
                # Older run_config.json has no worker count: recover effective
                # concurrency from cell durations over the span up to the last
                # completion.
                try:
                    span = (os.path.getmtime(
                        os.path.join(out_dir, "progress.jsonl"))
                        - (time.time() - elapsed))
                except OSError:
                    span = 0.0
                if span > 0 and mean_cell > 0:
                    concurrency = sum(cell_secs) / span
                    if concurrency > 0:
                        eta = remaining * mean_cell / concurrency
            if eta == float("inf") and done and elapsed > 0:
                eta = remaining * elapsed / done      # fallback
        L.append(f"  elapsed {_fmt_dur(elapsed):>7}    "
                 f"rate {rate_min:4.1f} cells/min    "
                 f"eta {_fmt_dur(eta):>7}"
                 + (f"  (~{time.strftime('%H:%M', time.localtime(time.time()+eta))})"
                    if eta not in (float('inf'),) else ""))
        L.append(f"  mean cell {mean_cell:6.1f}s    "
                 f"rows {rows:>11,}    "
                 f"on disk {_disk_bytes(out_dir)/1e9:5.2f} GB")
        free = _free_gb()
        last = recs[-1]
        L.append(f"  status {status:<10}"
                 + (f"   free RAM {free:4.1f} GB" if free is not None else ""))
        L.append("")
        L.append(f"  last: cell {last.get('cell_id'):3}  "
                 f"theta={last.get('theta_star')}  "
                 f"psi={last.get('psi_true')}  "
                 f"alpha={last.get('alpha')}  "
                 f"({last.get('wall_s')}s)")
    else:
        L.append("  no cells finished yet -- waiting for the first shard")
        L.append(f"  status {status}")

    # Completion grid: how many of the 11 alphas are done per (theta, psi).
    seen = {}
    for r in recs:
        key = (r.get("theta_star"), r.get("psi_true"))
        seen[key] = seen.get(key, 0) + 1
    L.append("")
    L.append("  alphas done per (theta*, psi), out of %d" % N_ALPHAS)
    L.append("        " + "".join(f"{p:>8}" for p in PSI_TRUES))
    for th in THETA_STARS:
        cells = []
        for psi in PSI_TRUES:
            c = seen.get((th, psi), 0)
            mark = "done" if c == N_ALPHAS else (f"{c}" if c else "-")
            cells.append(f"{mark:>8}")
        L.append(f"   {th:<4}" + "".join(cells))

    finished = status in ("complete", "partial", "incomplete") and done >= 0
    if status == "complete":
        L.append("")
        L.append("  SWEEP COMPLETE")
    elif status in ("partial", "incomplete"):
        L.append("")
        L.append(f"  SWEEP STOPPED EARLY ({status}): "
                 f"{cfg.get('stopped_early') or cfg.get('failed_cells')}")
        L.append("  rerun the same command to resume")
    return "\n".join(L), finished


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/full_sweep")
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    if args.once:
        screen, _ = render(args.out_dir)
        print(screen)
        return

    try:
        while True:
            screen, finished = render(args.out_dir)
            # Redraw from the top rather than scrolling.
            sys.stdout.write("\033[H\033[J" + screen + "\n")
            sys.stdout.flush()
            if finished:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n(viewer stopped; the sweep is unaffected)")


if __name__ == "__main__":
    main()
