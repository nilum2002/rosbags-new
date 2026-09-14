"""Step 7: render predictions against ground truth, one image per annotated frame.

Same drawing style as viz_rosbag/render.py. Each image shows, inside the camera's
110 deg sector (camera looking along +x):

  grey dots        laser returns the detector saw
  blue circles     ground truth (annotated people), with the association distance
                   drawn as a dashed ring and labelled gt<id>
  green dots       true positives: track states matched to a ground-truth person
  red crosses      false positives: track states with no person within the distance
  yellow squares   missed people (false negatives)

Only annotated frames are rendered, meaning scans with at least one annotated
person. Track states outside the camera FOV are dropped first, as in scoring.

Matching per frame is one-to-one within viz.association_distance (Hungarian on
distance). score_mot.py additionally keeps a track paired with the person it
matched on the previous scan, so on frames where two people are close the
image can differ from the score table by an observation or two. The table is
the official count.

Writes figures/<bag>/<detector>_<tracker>_conf<c>/frame_<scan>.png, where <scan> is
the scan's index within its bag, plus contact_sheet.png. Existing images in a
folder are replaced.
"""
import argparse
import glob
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Wedge  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from common import (DEFAULT_CONFIG, gt_file, inside_fov, load_config, load_npz,  # noqa: E402
                    preprocess, scans_file, sector_grid, tracks_file)

C_SCAN, C_GT, C_TP, C_FP, C_FN = "#9aa3ad", "#1f6feb", "#12a150", "#e5484d", "#f5a524"

_S = {}


def _init(state):
    _S.update(state)


def classify(track_xy, gt_xy, ad):
    """One-to-one match within `ad`. Returns (tp_mask over tracks, fn_mask over gt)."""
    tp = np.zeros(len(track_xy), dtype=bool)
    fn = np.ones(len(gt_xy), dtype=bool)
    if len(track_xy) and len(gt_xy):
        d = np.hypot(track_xy[:, None, 0] - gt_xy[None, :, 0], track_xy[:, None, 1] - gt_xy[None, :, 1])
        cost = np.where(d <= ad, d, 1e6)
        ti, gi = linear_sum_assignment(cost)
        ok = d[ti, gi] <= ad
        tp[ti[ok]] = True
        fn[gi[ok]] = False
    return tp, fn


def _draw(scan_i):
    S = _S
    grid, half, R, ad = S["grid"], S["half_deg"], S["range_max"], S["ad"]
    s = preprocess(S["scans"][scan_i])
    real = s < 25.0
    g_xy, g_id = S["gt"][scan_i]
    t_id, t_xy = S["tracks"].get(scan_i, (np.zeros(0, dtype=np.int64), np.zeros((0, 2))))
    tp, fn = classify(t_xy, g_xy, ad)

    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=110)
    ax.add_patch(Wedge((0, 0), R, -half, half, facecolor="#f2f4f7", edgecolor="#c9d1d9", lw=1.0, zorder=0))
    ax.scatter((s * np.cos(grid))[real], (s * np.sin(grid))[real], s=7, c=C_SCAN, zorder=2)

    for (x, y), gid in zip(g_xy, g_id):
        ax.add_patch(Circle((x, y), ad, fill=False, ec=C_GT, lw=1.4, ls="--", alpha=0.85, zorder=3))
        ax.plot(x, y, "o", ms=11, mfc="none", mec=C_GT, mew=2.2, zorder=4)
        ax.annotate(f"gt{int(gid)}", (x, y), textcoords="offset points", xytext=(10, 8),
                    color=C_GT, fontsize=8, fontweight="bold")
    if tp.any():
        ax.plot(t_xy[tp, 0], t_xy[tp, 1], "o", ms=7, color=C_TP, zorder=5)
    if (~tp).any():
        ax.plot(t_xy[~tp, 0], t_xy[~tp, 1], "X", ms=9, color=C_FP, zorder=5)
    if fn.any():
        ax.plot(g_xy[fn, 0], g_xy[fn, 1], "s", ms=9, mfc="none", mec=C_FN, mew=2.2, zorder=5)
    for (x, y), tid in zip(t_xy, t_id):
        ax.annotate(f"t{int(tid)}", (x, y), textcoords="offset points", xytext=(8, -12),
                    color="#24292f", fontsize=8)

    ax.plot(0, 0, "^", ms=13, color="#24292f", zorder=6)
    ax.set_xlim(-0.6, R)
    ax.set_ylim(-R * 0.72, R * 0.72)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlabel("x [m] (camera forward)")
    ax.set_ylabel("y [m]")
    n_tp, n_fp, n_fn = int(tp.sum()), int((~tp).sum()), int(fn.sum())
    ax.set_title(f"{S['bag']}   scan {scan_i}\n{S['detector']} + {S['tracker']}   conf={S['conf']}   "
                 f"TP {n_tp}  FP {n_fp}  FN {n_fn}", fontsize=10)
    handles = [plt.Line2D([], [], ls="", marker=mk, color=c, mfc=mfc, mec=c, ms=8, label=lab)
               for mk, c, mfc, lab in [("o", C_SCAN, C_SCAN, "laser return"),
                                       ("o", C_GT, "none", "ground truth"),
                                       ("o", C_TP, C_TP, "true positive"),
                                       ("X", C_FP, C_FP, "false positive"),
                                       ("s", C_FN, "none", "missed (FN)")]]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.92)
    pose = S["poses"].get(scan_i)
    if pose:
        ax.text(0.01, 0.01, f"robot odom  x={pose['x']:+.2f}  y={pose['y']:+.2f}  "
                            f"yaw={math.degrees(pose['yaw']):+.1f}°",
                transform=ax.transAxes, fontsize=7.5, color="#57606a")
    fig.tight_layout()
    fig.savefig(os.path.join(S["out_dir"], f"frame_{scan_i:05d}.png"))
    plt.close(fig)
    return n_tp, n_fp, n_fn


def contact_sheet(paths, out):
    n = len(paths)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.2), dpi=110)
    axes = np.atleast_1d(axes).ravel()
    for a in axes:
        a.axis("off")
    for a, p in zip(axes, paths):
        a.imshow(plt.imread(p))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--bag", action="append", help="limit to these bags")
    ap.add_argument("--detector", action="append", help="limit to these detector names")
    ap.add_argument("--tracker", action="append", help="limit to these trackers")
    ap.add_argument("--conf", type=float, action="append", help="limit to these thresholds")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    viz, p = cfg["viz"], cfg["paths"]
    grid = sector_grid(cfg)
    half = cfg["sensor"]["fov_half_deg"]
    ad = float(viz.get("association_distance", 0.5))
    gate = bool(cfg["scoring"].get("fov_gate", True))
    gt = load_npz(gt_file(cfg))
    names = [str(n) for n in gt["bag_names"]]
    rsi = gt["row_scan_index"]

    for det in cfg["detectors"]:
        if args.detector and det["name"] not in args.detector:
            continue
        for trk in cfg["trackers"]:
            if args.tracker and trk not in args.tracker:
                continue
            for conf in cfg["conf_thresholds"]:
                if args.conf and conf not in args.conf:
                    continue
                tr = load_npz(tracks_file(cfg, det["name"], trk, conf))
                for s, bag in enumerate(names):
                    if args.bag and bag not in args.bag:
                        continue
                    out_dir = os.path.join(p["figures_dir"], bag, f"{det['name']}_{trk}_conf{conf}")
                    os.makedirs(out_dir, exist_ok=True)
                    for old in glob.glob(os.path.join(out_dir, "*.png")):
                        os.remove(old)

                    rows = np.nonzero(gt["segment"] == s)[0]
                    g = np.isin(gt["gt_scan"], rows)
                    g_scan = rsi[gt["gt_scan"][g]]
                    gt_by_scan = {int(sc): (gt["gt_xy"][g][g_scan == sc], gt["gt_id"][g][g_scan == sc])
                                  for sc in np.unique(g_scan)}
                    frames = sorted(gt_by_scan)  # annotated frames: at least one person

                    m = tr["segment"] == s
                    t_row, t_id, t_xy = tr["row"][m], tr["track_id"][m], tr["xy"][m]
                    if gate and len(t_xy):
                        k = inside_fov(t_xy, half)
                        t_row, t_id, t_xy = t_row[k], t_id[k], t_xy[k]
                    t_scan = rsi[t_row] if len(t_row) else np.zeros(0, dtype=np.int64)
                    tracks = {int(sc): (t_id[t_scan == sc], t_xy[t_scan == sc]) for sc in np.unique(t_scan)}

                    with open(os.path.join(p["bags_dir"], cfg["bags"][s]["annotations"])) as f:
                        poses = {int(k): v.get("pose") for k, v in json.load(f)["frames"].items()}

                    state = dict(grid=grid, half_deg=half, range_max=float(viz.get("range_max_m", 5.0)), ad=ad,
                                 scans=load_npz(scans_file(cfg, bag))["scans"], gt=gt_by_scan, tracks=tracks,
                                 poses=poses, conf=conf, bag=bag, detector=det["name"], tracker=trk,
                                 out_dir=out_dir)
                    with Pool(int(viz.get("workers", 8)), initializer=_init, initargs=(state,)) as pool:
                        counts = np.array(pool.map(_draw, frames, chunksize=16)).reshape(-1, 3)

                    n_sheet = int(viz.get("contact_sheet_frames", 16))
                    if n_sheet and frames:
                        pick = [frames[j] for j in
                                np.linspace(0, len(frames) - 1, min(n_sheet, len(frames))).astype(int)]
                        contact_sheet([os.path.join(out_dir, f"frame_{i:05d}.png") for i in pick],
                                      os.path.join(out_dir, "contact_sheet.png"))
                    tp, fp, fn = counts.sum(axis=0) if len(counts) else (0, 0, 0)
                    print(f"  {bag:<13} {det['name']:<38} {trk:<8} conf={conf:<5} {len(frames)} annotated frames  "
                          f"TP {tp}  FP {fp}  FN {fn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
