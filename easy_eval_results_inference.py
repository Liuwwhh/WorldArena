#!/usr/bin/env python3
"""
One-click evaluation entry for `results_inference`.

What this script does:
1) Read `sample_records.jsonl` and find stitched videos under `videos/`.
2) Split each stitched video into GT (left column) and generated (right column).
3) Build WorldArena-compatible `summary.json` and directory layout automatically.
4) Optionally run standard metrics / action_following / VLM / JEPA.
5) Aggregate available results into one CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import yaml


AUTO_METRICS_ORDER = [
    "psnr",
    "ssim",
    "mse",
    "lpips",
    "fid",
    "fvd",
    "trajectory_accuracy",
    "semantic_alignment",
    "depth_accuracy",
    "aesthetic_quality",
    "background_consistency",
    "dynamic_degree",
    "flow_score",
    "photometric_smoothness",
    "motion_smoothness",
    "image_quality",
    "subject_consistency",
    # "action_following",
]
GLOBAL_DATASET_METRICS = {"fid", "fvd"}


@dataclass
class PreparedSample:
    video_id: str
    gt_video: Path
    gen_standard_video: Path
    gen_vlm_video: Path
    gen_jepa_video: Path
    gt_jepa_video: Path
    first_frame: Path
    prompt: str


def log(msg: str) -> None:
    print(f"[easy-eval] {msg}")


def run_cmd(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    allow_failure: bool = False,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
) -> int:
    cmd_str = " ".join(str(x) for x in cmd)
    log(f"Run: {cmd_str}")
    if log_path is None:
        proc = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(cwd) if cwd else None,
            env=env,
        )
    else:
        ensure_dir(log_path.parent)
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.run(
                [str(x) for x in cmd],
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
    if proc.returncode != 0 and not allow_failure:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd_str}")
    if proc.returncode != 0:
        log(f"[WARN] Command failed but ignored ({proc.returncode}): {cmd_str}")
    elif log_path is not None:
        log(f"Done: {cmd_str} (log: {log_path})")
    return proc.returncode


def build_subprocess_cmd(
    script_path: Path,
    args: argparse.Namespace,
    phase_flag: Optional[str],
) -> List[str]:
    cmd: List[str] = [
        args.conda_bin,
        "run",
        "-n",
        args.env_base,
        "python",
        str(script_path),
        "--results_dir",
        str(args.results_dir.resolve()),
        "--model_name",
        args.model_name,
        "--task_name",
        args.task_name,
        "--row_mode",
        args.row_mode,
        "--metrics",
        args.metrics,
        "--exclude_metrics",
        args.exclude_metrics,
        "--vlm_num_frames",
        str(args.vlm_num_frames),
        "--csv_name",
        args.csv_name,
    ]
    if args.work_dir is not None:
        cmd.extend(["--work_dir", str(args.work_dir.resolve())])
    if args.base_config is not None:
        cmd.extend(["--base_config", str(args.base_config.resolve())])
    if args.records_jsonl is not None:
        cmd.extend(["--records_jsonl", str(args.records_jsonl.resolve())])
    if args.videos_dir is not None:
        cmd.extend(["--videos_dir", str(args.videos_dir.resolve())])
    if args.force_rebuild:
        cmd.append("--force_rebuild")
    if args.allow_failure:
        cmd.append("--allow_failure")
    if args.prepare_only:
        cmd.append("--prepare_only")
    if args.skip_aggregate:
        cmd.append("--skip_aggregate")
    if args.resize_generated:
        cmd.append("--resize_generated")
    if phase_flag:
        cmd.append(phase_flag)
    return cmd


def dispatch_multi_env(args: argparse.Namespace, script_path: Path) -> None:
    if args.prepare_only:
        cmd = build_subprocess_cmd(script_path, args, None)
        cmd[3] = args.env_base
        run_cmd(cmd, allow_failure=args.allow_failure)
        return

    jobs: List[Tuple[str, str]] = []
    if args.run_standard:
        jobs.append((args.env_base, "--run_standard"))
    if args.run_action_following:
        jobs.append((args.env_base, "--run_action_following"))
    if args.run_vlm:
        jobs.append((args.env_vlm, "--run_vlm"))
    if args.run_jepa:
        jobs.append((args.env_jepa, "--run_jepa"))

    if not jobs:
        jobs.append((args.env_base, "--run_standard"))

    job_desc = ", ".join([f"{flag}@{env}" for env, flag in jobs])
    log(f"Auto env switch enabled. Dispatch jobs: {job_desc}")

    for env_name, phase_flag in jobs:
        cmd = build_subprocess_cmd(script_path, args, phase_flag)
        cmd[3] = env_name
        run_cmd(cmd, allow_failure=args.allow_failure)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_placeholder(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    v = value.strip().lower()
    return (not v) or ("your absolute path" in v) or (v in {"none", "null"})


def deep_get(mapping: Dict, keys: Sequence[str]) -> object:
    cur: object = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                log(f"[WARN] Skip invalid JSON line {line_no}: {path}")
    return records


def resolve_input_video(record_video_path: str, local_videos_dir: Path) -> Optional[Path]:
    rec_path = Path(record_video_path)
    by_name = local_videos_dir / rec_path.name
    if by_name.exists():
        return by_name
    if rec_path.exists():
        return rec_path
    return None


def parse_episode_base(record: Dict, video_name: str) -> str:
    ep_idx = record.get("episode_index")
    if ep_idx is not None:
        ep_str = str(ep_idx)
        if ep_str.startswith("episode"):
            return ep_str
        return f"episode{ep_str}"
    match = re.search(r"ep(\d+)", video_name)
    if match:
        return f"episode{match.group(1)}"
    return "episode_unknown"


def make_video_id(record: Dict, video_name: str, fallback_index: int) -> str:
    ep = parse_episode_base(record, video_name)
    sample_idx = record.get("sample_index", fallback_index)
    try:
        sample_idx_int = int(sample_idx)
    except (TypeError, ValueError):
        sample_idx_int = fallback_index
    return f"{ep}_s{sample_idx_int:06d}"


def row_bounds(height: int, row_mode: str) -> Tuple[int, int]:
    if row_mode == "full":
        return 0, height

    idx_map = {"top": 0, "middle": 1, "bottom": 2}
    row_idx = idx_map[row_mode]
    row_h = max(1, height // 3)
    y0 = row_idx * row_h
    y1 = height if row_idx == 2 else min(height, (row_idx + 1) * row_h)
    return y0, y1


def split_stitched_video(
    src_video: Path,
    gt_video_out: Path,
    gen_video_out: Path,
    first_frame_out: Path,
    row_mode: str,
) -> int:
    ensure_dir(gt_video_out.parent)
    ensure_dir(gen_video_out.parent)
    ensure_dir(first_frame_out.parent)

    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {src_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 2 or height < 2:
        cap.release()
        raise RuntimeError(f"Invalid video shape ({width}x{height}): {src_video}")

    split_x = width // 2
    y0, y1 = row_bounds(height, row_mode)

    out_w = split_x
    out_h = max(1, y1 - y0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    gt_writer = cv2.VideoWriter(str(gt_video_out), fourcc, fps, (out_w, out_h))
    gen_writer = cv2.VideoWriter(str(gen_video_out), fourcc, fps, (out_w, out_h))

    n = 0
    first_saved = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue

        gt_frame = frame[y0:y1, :split_x]
        gen_frame = frame[y0:y1, split_x:width]
        if gt_frame.size == 0 or gen_frame.size == 0:
            continue

        gt_writer.write(gt_frame)
        gen_writer.write(gen_frame)

        if not first_saved:
            cv2.imwrite(str(first_frame_out), gt_frame)
            first_saved = True
        n += 1

    cap.release()
    gt_writer.release()
    gen_writer.release()

    if n == 0:
        raise RuntimeError(f"No frames written after split: {src_video}")
    return n


def copy_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def duplicate_test_variants(gen_test_dir: Path) -> None:
    for suffix in ("_1", "_2"):
        dst_dir = gen_test_dir.parent / f"{gen_test_dir.name}{suffix}"
        ensure_dir(dst_dir)
        for src in sorted(gen_test_dir.glob("*.mp4")):
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)


def load_base_ckpt(base_config_path: Optional[Path]) -> Dict:
    if not base_config_path or not base_config_path.exists():
        return {}
    try:
        with base_config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if isinstance(cfg, dict):
            ckpt = cfg.get("ckpt", {})
            if isinstance(ckpt, dict):
                return ckpt
    except Exception as exc:  # noqa: BLE001
        log(f"[WARN] Failed to load base config `{base_config_path}`: {exc}")
    return {}


def resolve_config_path(base_config_path: Optional[Path], project_root: Path) -> Optional[Path]:
    if base_config_path is None:
        return None

    candidate = Path(base_config_path)
    if candidate.is_absolute():
        return candidate

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = (project_root / candidate).resolve()
    if project_candidate.exists():
        return project_candidate

    # keep deterministic absolute path even when file does not exist
    return project_candidate


def make_auto_config(work_dir: Path, model_name: str, ckpt: Dict) -> Dict:
    return {
        "model_name": model_name,
        "data": {
            "gt_path": str((work_dir / "data" / "gt_dataset").resolve()),
            "val_base": str((work_dir / "data" / "generated_dataset").resolve()),
        },
        "data_action_following": {
            "gt_path": str((work_dir / "data_action_following" / "gt_dataset").resolve()),
            "val_base": str((work_dir / "data_action_following" / "generated_dataset").resolve()),
        },
        "save_path": str((work_dir / "output").resolve()),
        "save_path_action_following": str((work_dir / "output_action_following").resolve()),
        "ckpt": ckpt,
    }


def write_yaml(path: Path, data: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def required_ckpt_ok(metric: str, cfg: Dict) -> bool:
    if metric in {"psnr", "ssim", "mse"}:
        return True
    if metric == "trajectory_accuracy":
        sam3_path = deep_get(cfg, ["ckpt", "sam3_model_ckpt"])
        return not is_placeholder(sam3_path)

    req: Dict[str, List[Sequence[str]]] = {
        "semantic_alignment": [
            ("ckpt", "semantic_alignment", "caption"),
            ("ckpt", "semantic_alignment", "CLIP"),
        ],
        "depth_accuracy": [("ckpt", "depth_accuracy")],
        "aesthetic_quality": [
            ("ckpt", "aesthetic_quality", "clip"),
            ("ckpt", "aesthetic_quality", "aesthetic_head"),
        ],
        "background_consistency": [
            ("ckpt", "background_consistency", "clip"),
            ("ckpt", "background_consistency", "raft"),
        ],
        "dynamic_degree": [("ckpt", "dynamic_degree", "raft")],
        "flow_score": [("ckpt", "flow_score", "raft")],
        "photometric_smoothness": [
            ("ckpt", "photometric_smoothness", "cfg"),
            ("ckpt", "photometric_smoothness", "model"),
        ],
        "motion_smoothness": [("ckpt", "motion_smoothness", "model")],
        "image_quality": [("ckpt", "image_quality", "musiq")],
        "subject_consistency": [
            ("ckpt", "subject_consistency", "repo"),
            ("ckpt", "subject_consistency", "weight"),
            ("ckpt", "subject_consistency", "raft"),
        ],
        "action_following": [("ckpt", "action_following")],
        "lpips": [("ckpt", "lpips", "alexnet")],
        "fid": [("ckpt", "fid", "inception")],
        "fvd": [("ckpt", "fvd", "i3d")],
    }
    if metric not in req:
        return True
    for key_path in req[metric]:
        value = deep_get(cfg, key_path)
        if is_placeholder(value):
            return False
    return True


def select_metrics(metrics_arg: str, cfg: Dict) -> List[str]:
    if metrics_arg.strip().lower() == "auto":
        requested = AUTO_METRICS_ORDER
    else:
        requested = [m.strip() for m in metrics_arg.split(",") if m.strip()]

    selected: List[str] = []
    for metric in requested:
        if required_ckpt_ok(metric, cfg):
            selected.append(metric)
        else:
            log(f"[WARN] Skip metric `{metric}` because required ckpt is missing/placeholder.")
    return selected


def parse_metric_csv(metrics: str) -> List[str]:
    if not metrics:
        return []
    return [m.strip() for m in metrics.split(",") if m.strip()]


def apply_metric_exclusion(metrics: List[str], exclude_metrics: str) -> List[str]:
    excludes = set(parse_metric_csv(exclude_metrics))
    if not excludes:
        return metrics
    return [m for m in metrics if m not in excludes]


def resolve_inputs(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    results_dir = args.results_dir.resolve()
    videos_dir = args.videos_dir.resolve() if args.videos_dir else (results_dir / "videos")
    records_path = args.records_jsonl.resolve() if args.records_jsonl else (results_dir / "sample_records.jsonl")
    return results_dir, videos_dir, records_path


def save_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_records_round_robin(records: Sequence[Dict], n_shards: int) -> List[List[Dict]]:
    shards: List[List[Dict]] = [[] for _ in range(n_shards)]
    for idx, rec in enumerate(records):
        shards[idx % n_shards].append(rec)
    return shards


def parse_gpu_ids(gpu_ids_arg: str) -> List[str]:
    raw = gpu_ids_arg.strip() if gpu_ids_arg else ""
    if not raw:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return ["0"]

    dedup: List[str] = []
    seen = set()
    for item in raw.split(","):
        gid = item.strip()
        if not gid:
            continue
        if gid in seen:
            continue
        seen.add(gid)
        dedup.append(gid)
    return dedup or ["0"]


def tail_text(path: Path, n_lines: int = 80) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n_lines:])


def deep_merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if key in dst and isinstance(dst[key], dict) and isinstance(value, dict):
            deep_merge_dict(dst[key], value)
        else:
            dst[key] = value
    return dst


def merge_list_metric_payload(payloads: Sequence[Any]) -> Any:
    list_payloads = [p for p in payloads if isinstance(p, list)]
    if not list_payloads:
        return payloads[0] if payloads else None

    entry_payloads = [p for p in list_payloads if len(p) >= 2 and isinstance(p[1], list)]
    if not entry_payloads:
        return list_payloads[0]

    merged_entries: List[Dict[str, Any]] = []
    for payload in entry_payloads:
        merged_entries.extend(payload[1])

    vals: List[float] = []
    for item in merged_entries:
        if not isinstance(item, dict):
            continue
        val = item.get("video_results")
        if isinstance(val, (int, float)):
            vals.append(float(val))
    overall = float(sum(vals) / len(vals)) if vals else 0.0
    return [overall, merged_entries]


def merge_result_json_files(
    input_jsons: Sequence[Path],
    output_json: Path,
    global_json: Optional[Path] = None,
    global_metrics: Optional[Iterable[str]] = None,
) -> None:
    metrics_global = set(global_metrics or [])
    json_dicts: List[Dict[str, Any]] = []
    metric_order: List[str] = []
    seen_metric = set()

    for path in input_jsons:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        json_dicts.append(data)
        for key in data.keys():
            if key not in seen_metric:
                seen_metric.add(key)
                metric_order.append(key)

    global_data: Dict[str, Any] = {}
    if global_json and global_json.exists():
        with global_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            global_data = data
            for key in global_data.keys():
                if key not in seen_metric:
                    seen_metric.add(key)
                    metric_order.append(key)

    merged: Dict[str, Any] = {}
    for metric in metric_order:
        if metric in metrics_global and metric in global_data:
            merged[metric] = global_data[metric]
            continue

        payloads = [d[metric] for d in json_dicts if metric in d]
        if not payloads:
            if metric in global_data:
                merged[metric] = global_data[metric]
            continue

        first = payloads[0]
        if isinstance(first, list):
            merged[metric] = merge_list_metric_payload(payloads)
        elif isinstance(first, dict):
            cur: Dict[str, Any] = {}
            for payload in payloads:
                if isinstance(payload, dict):
                    deep_merge_dict(cur, payload)
            merged[metric] = cur
        else:
            merged[metric] = first

    ensure_dir(output_json.parent)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def build_child_eval_cmd(
    args: argparse.Namespace,
    script_path: Path,
    work_dir: Path,
    metrics_str: str,
    run_standard: bool = False,
    run_action_following: bool = False,
    run_vlm: bool = False,
    run_jepa: bool = False,
    env_name: Optional[str] = None,
    records_jsonl: Optional[Path] = None,
    videos_dir: Optional[Path] = None,
) -> List[str]:
    env_to_use = env_name or args.env_base
    cmd: List[str] = [
        args.conda_bin,
        "run",
        "-n",
        env_to_use,
        "python",
        str(script_path),
        "--results_dir",
        str(args.results_dir.resolve()),
        "--work_dir",
        str(work_dir.resolve()),
        "--model_name",
        args.model_name,
        "--task_name",
        args.task_name,
        "--row_mode",
        args.row_mode,
        "--metrics",
        metrics_str,
        "--exclude_metrics",
        "",
        "--vlm_num_frames",
        str(args.vlm_num_frames),
        "--csv_name",
        args.csv_name,
        "--skip_aggregate",
    ]
    if args.base_config is not None:
        cmd.extend(["--base_config", str(args.base_config.resolve())])
    if records_jsonl is not None:
        cmd.extend(["--records_jsonl", str(records_jsonl.resolve())])
    if videos_dir is not None:
        cmd.extend(["--videos_dir", str(videos_dir.resolve())])

    if run_standard:
        cmd.append("--run_standard")
    if run_action_following:
        cmd.append("--run_action_following")
    if run_vlm:
        cmd.append("--run_vlm")
    if run_jepa:
        cmd.append("--run_jepa")
    if args.force_rebuild:
        cmd.append("--force_rebuild")
    if args.allow_failure:
        cmd.append("--allow_failure")
    if args.resize_generated:
        cmd.append("--resize_generated")
    return cmd


def run_multi_gpu(args: argparse.Namespace, script_path: Path, project_root: Path) -> bool:
    if args.auto_env_switch:
        raise ValueError("`--multi_gpu` is not compatible with `--auto_env_switch`. Please disable one of them.")
    if args.prepare_only:
        log("[WARN] `--multi_gpu` with `--prepare_only` falls back to single-process prepare.")
        return False
    if not args.run_standard and not args.run_action_following:
        log("[WARN] `--multi_gpu` only accelerates standard/action_following phases. Falling back to single-process.")
        return False

    results_dir, videos_dir, records_path = resolve_inputs(args)
    if not videos_dir.exists() or not records_path.exists():
        raise FileNotFoundError(
            f"`{results_dir}` must provide valid videos and records: videos={videos_dir}, records={records_path}"
        )
    records = load_jsonl(records_path)
    if not records:
        raise RuntimeError(f"No valid records found in {records_path}")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    n_workers = min(len(gpu_ids), len(records))
    if n_workers <= 1:
        log("[WARN] `--multi_gpu` requested but only one worker is available. Falling back to single-process.")
        return False

    if args.base_config and not args.base_config.exists():
        log(f"[WARN] base config not found: {args.base_config}")
    ckpt = load_base_ckpt(args.base_config if args.base_config else None)
    selected_metrics = select_metrics(args.metrics, {"ckpt": ckpt})
    selected_metrics = apply_metric_exclusion(selected_metrics, args.exclude_metrics)

    selected_standard = [m for m in selected_metrics if m != "action_following"]
    standard_for_workers = [m for m in selected_standard if m not in GLOBAL_DATASET_METRICS]
    standard_global_only = [m for m in selected_standard if m in GLOBAL_DATASET_METRICS]
    action_selected = "action_following" in selected_metrics

    worker_metrics: List[str] = []
    if args.run_standard:
        worker_metrics.extend(standard_for_workers)
    if args.run_action_following and action_selected:
        worker_metrics.append("action_following")

    # stable de-dup
    seen = set()
    worker_metrics = [m for m in worker_metrics if not (m in seen or seen.add(m))]

    work_dir = args.work_dir.resolve() if args.work_dir else (results_dir / "worldarena_auto_eval").resolve()
    ensure_dir(work_dir)
    shard_root = work_dir / "_multi_gpu_shards"
    ensure_dir(shard_root)

    log(f"[multi-gpu] results_dir: {results_dir}")
    log(f"[multi-gpu] work_dir: {work_dir}")
    log(f"[multi-gpu] records: {len(records)}, workers: {n_workers}, gpu_ids: {gpu_ids[:n_workers]}")
    log(f"[multi-gpu] selected metrics: {selected_metrics}")
    log(f"[multi-gpu] worker metrics: {worker_metrics}")
    if standard_global_only:
        log(f"[multi-gpu] global-only metrics (single full pass): {standard_global_only}")

    shards = split_records_round_robin(records, n_workers)

    worker_jobs = []
    if worker_metrics:
        worker_metrics_str = ",".join(worker_metrics)
        for shard_id, shard_records in enumerate(shards):
            if not shard_records:
                continue
            shard_dir = shard_root / f"shard_{shard_id:02d}"
            ensure_dir(shard_dir)
            shard_records_path = shard_dir / "sample_records.jsonl"
            save_jsonl(shard_records_path, shard_records)
            shard_work = shard_dir / "work"

            run_worker_standard = args.run_standard and bool(standard_for_workers)
            run_worker_action = args.run_action_following and action_selected
            if not run_worker_standard and not run_worker_action:
                continue

            cmd = build_child_eval_cmd(
                args=args,
                script_path=script_path,
                work_dir=shard_work,
                metrics_str=worker_metrics_str,
                run_standard=run_worker_standard,
                run_action_following=run_worker_action,
                records_jsonl=shard_records_path,
                videos_dir=videos_dir,
                env_name=args.env_base,
            )
            log_path = shard_dir / "worker.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids[shard_id % n_workers]
            worker_jobs.append(
                {
                    "shard_id": shard_id,
                    "gpu": env["CUDA_VISIBLE_DEVICES"],
                    "cmd": cmd,
                    "env": env,
                    "log_path": log_path,
                    "work_dir": shard_work,
                }
            )

    failed_jobs = []
    successful_jobs = []
    running = []
    for job in worker_jobs:
        log(f"[multi-gpu] start worker shard={job['shard_id']} gpu={job['gpu']}")
        ensure_dir(job["log_path"].parent)
        f = job["log_path"].open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [str(x) for x in job["cmd"]],
            env=job["env"],
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        running.append((job, proc, f))

    for job, proc, f in running:
        rc = proc.wait()
        f.close()
        if rc != 0:
            failed_jobs.append((job, rc))
            log(f"[multi-gpu][ERROR] worker shard={job['shard_id']} failed (rc={rc}), log={job['log_path']}")
        else:
            successful_jobs.append(job)
            log(f"[multi-gpu] worker shard={job['shard_id']} done, log={job['log_path']}")

    if failed_jobs and not args.allow_failure:
        msg_parts = []
        for job, rc in failed_jobs[:3]:
            tail = tail_text(job["log_path"], n_lines=60)
            msg_parts.append(
                f"shard={job['shard_id']} gpu={job['gpu']} rc={rc}\nlog={job['log_path']}\n--- tail ---\n{tail}"
            )
        raise RuntimeError(
            "One or more multi-gpu workers failed.\n\n" + "\n\n".join(msg_parts)
        )

    global_metrics_json: Optional[Path] = None
    if args.run_standard and standard_global_only:
        global_dir = shard_root / "global_standard"
        global_log = global_dir / "worker.log"
        global_cmd = build_child_eval_cmd(
            args=args,
            script_path=script_path,
            work_dir=global_dir / "work",
            metrics_str=",".join(standard_global_only),
            run_standard=True,
            run_action_following=False,
            records_jsonl=records_path,
            videos_dir=videos_dir,
            env_name=args.env_base,
        )
        global_env = os.environ.copy()
        global_env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        run_cmd(global_cmd, allow_failure=args.allow_failure, env=global_env, log_path=global_log)
        candidate = global_dir / "work" / "output" / "generated_results.json"
        if candidate.exists():
            global_metrics_json = candidate
        else:
            msg = f"Global metrics json not found: {candidate}"
            if args.allow_failure:
                log(f"[WARN] {msg}")
            else:
                raise RuntimeError(msg)

    # Merge shard outputs -> final work_dir outputs
    if args.run_standard:
        shard_standard_jsons = [
            Path(job["work_dir"]) / "output" / "generated_results.json"
            for job in successful_jobs
            if (Path(job["work_dir"]) / "output" / "generated_results.json").exists()
        ]
        if shard_standard_jsons or global_metrics_json:
            merged_standard_json = work_dir / "output" / "generated_results.json"
            merge_result_json_files(
                input_jsons=shard_standard_jsons,
                output_json=merged_standard_json,
                global_json=global_metrics_json,
                global_metrics=standard_global_only,
            )
            log(f"[multi-gpu] merged standard results: {merged_standard_json}")
        else:
            msg = "No standard result json found from workers."
            if args.allow_failure:
                log(f"[WARN] {msg}")
            else:
                raise RuntimeError(msg)

    if args.run_action_following and action_selected:
        shard_action_jsons = [
            Path(job["work_dir"]) / "output_action_following" / "generated_results.json"
            for job in successful_jobs
            if (Path(job["work_dir"]) / "output_action_following" / "generated_results.json").exists()
        ]
        if shard_action_jsons:
            merged_action_json = work_dir / "output_action_following" / "generated_results.json"
            merge_result_json_files(
                input_jsons=shard_action_jsons,
                output_json=merged_action_json,
            )
            log(f"[multi-gpu] merged action_following results: {merged_action_json}")
        else:
            log("[WARN] No action_following result json found from workers.")

    # Run VLM / JEPA as single full pass (no sharding)
    if args.run_vlm:
        vlm_dir = shard_root / "vlm_full"
        vlm_cmd = build_child_eval_cmd(
            args=args,
            script_path=script_path,
            work_dir=work_dir,
            metrics_str=args.metrics,
            run_vlm=True,
            env_name=args.env_vlm,
            records_jsonl=records_path,
            videos_dir=videos_dir,
        )
        vlm_env = os.environ.copy()
        vlm_env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        run_cmd(vlm_cmd, allow_failure=args.allow_failure, env=vlm_env, log_path=vlm_dir / "worker.log")

    if args.run_jepa:
        jepa_dir = shard_root / "jepa_full"
        jepa_cmd = build_child_eval_cmd(
            args=args,
            script_path=script_path,
            work_dir=work_dir,
            metrics_str=args.metrics,
            run_jepa=True,
            env_name=args.env_jepa,
            records_jsonl=records_path,
            videos_dir=videos_dir,
        )
        jepa_env = os.environ.copy()
        jepa_env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        run_cmd(jepa_cmd, allow_failure=args.allow_failure, env=jepa_env, log_path=jepa_dir / "worker.log")

    # Keep a readable config in final work dir for reproducibility.
    final_auto_cfg = make_auto_config(work_dir, args.model_name, ckpt)
    write_yaml(work_dir / "config.auto.yaml", final_auto_cfg)

    # Aggregate once after all phases finish.
    if not args.skip_aggregate:
        aggregate_cmd = [
            args.conda_bin,
            "run",
            "-n",
            args.env_base,
            "python",
            str(project_root / "video_quality" / "csv_results" / "aggregate_results.py"),
            "--base_dir",
            str(work_dir),
            "--model_name",
            args.model_name,
            "--csv_name",
            args.csv_name,
        ]
        run_cmd(
            aggregate_cmd,
            allow_failure=args.allow_failure,
            log_path=shard_root / "aggregate.log",
        )

        csv_path = work_dir / "csv_results" / args.csv_name
        if csv_path.exists():
            log(f"[multi-gpu] done. Aggregated CSV: {csv_path}")
        else:
            log(f"[multi-gpu][WARN] done, but CSV not found: {csv_path}")
    else:
        log("[multi-gpu] done. Skip aggregation by `--skip_aggregate`.")
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Easy WorldArena evaluation directly from `results_inference`."
    )
    p.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Path to results_inference (must contain videos/ and sample_records.jsonl).",
    )
    p.add_argument(
        "--work_dir",
        type=Path,
        default=None,
        help="Workspace for auto-prepared files and outputs. Default: <results_dir>/worldarena_auto_eval",
    )
    p.add_argument("--model_name", type=str, default="results_inference_auto")
    p.add_argument("--task_name", type=str, default="task_auto")
    p.add_argument(
        "--row_mode",
        type=str,
        default="full",
        choices=["full", "top", "middle", "bottom"],
        help="If stitched video has 3 rows, choose which row to evaluate. Default full column.",
    )
    p.add_argument(
        "--metrics",
        type=str,
        default="auto",
        help="Comma list. Use `auto` to select metrics with available ckpts.",
    )
    p.add_argument(
        "--exclude_metrics",
        type=str,
        default="",
        help="Comma list of metrics to exclude after selection.",
    )
    p.add_argument(
        "--base_config",
        type=Path,
        default=Path("video_quality/config/config.yaml"),
        help="Optional config used to import ckpt paths.",
    )
    p.add_argument("--run_standard", action="store_true", help="Run standard metrics.")
    p.add_argument("--run_action_following", action="store_true", help="Run action_following metric.")
    p.add_argument("--run_vlm", action="store_true", help="Run VLM metrics.")
    p.add_argument("--run_jepa", action="store_true", help="Run JEPA metric.")
    p.add_argument("--prepare_only", action="store_true", help="Only prepare data, do not run evaluation.")
    p.add_argument("--skip_aggregate", action="store_true", help="Skip final CSV aggregation step.")
    p.add_argument(
        "--auto_env_switch",
        action="store_true",
        help=(
            "Dispatch phases to different conda envs automatically: "
            "standard/action_following -> env_base, VLM -> env_vlm, JEPA -> env_jepa."
        ),
    )
    p.add_argument(
        "--multi_gpu",
        action="store_true",
        help=(
            "Shard videos across multiple GPUs (one process per GPU, each process handles a subset of videos). "
            "Global metrics like FID/FVD are computed once on the full set."
        ),
    )
    p.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help="GPU ids for multi-gpu mode, e.g. `0,1,2,3`. Defaults to CUDA_VISIBLE_DEVICES.",
    )
    p.add_argument(
        "--records_jsonl",
        type=Path,
        default=None,
        help="Optional records jsonl path. Default: <results_dir>/sample_records.jsonl",
    )
    p.add_argument(
        "--videos_dir",
        type=Path,
        default=None,
        help="Optional videos directory. Default: <results_dir>/videos",
    )
    p.add_argument("--conda_bin", type=str, default="conda", help="Conda executable name/path.")
    p.add_argument("--env_base", type=str, default="WorldArena")
    p.add_argument("--env_vlm", type=str, default="WorldArena_VLM")
    p.add_argument("--env_jepa", type=str, default="WorldArena_JEPA")
    p.add_argument(
        "--resize_generated",
        action="store_true",
        help=(
            "Resize generated frame folders to 640x480 before standard metrics. "
            "Default disabled (keep original resolution)."
        ),
    )
    p.add_argument("--force_rebuild", action="store_true", help="Force re-split videos.")
    p.add_argument(
        "--allow_failure",
        action="store_true",
        help="Do not stop when an optional evaluation command fails.",
    )
    p.add_argument("--vlm_num_frames", type=int, default=16)
    p.add_argument("--csv_name", type=str, default="aggregated_results.csv")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    script_path = Path(__file__).resolve()
    project_root = script_path.parent
    args.base_config = resolve_config_path(args.base_config, project_root)

    if not args.prepare_only and not any(
        [args.run_standard, args.run_action_following, args.run_vlm, args.run_jepa]
    ):
        args.run_standard = True
        log("No evaluation switches provided; defaulting to standard metrics (`--run_standard`).")

    if args.multi_gpu:
        handled = run_multi_gpu(args=args, script_path=script_path, project_root=project_root)
        if handled:
            return

    if args.auto_env_switch:
        dispatch_multi_env(args, script_path)
        return

    video_quality_dir = project_root / "video_quality"
    py = sys.executable

    results_dir, videos_dir, records_path = resolve_inputs(args)
    if not videos_dir.exists() or not records_path.exists():
        raise FileNotFoundError(
            f"`{results_dir}` must contain `videos/` and `sample_records.jsonl` "
            f"(or provide --videos_dir/--records_jsonl)."
        )

    work_dir = args.work_dir.resolve() if args.work_dir else (results_dir / "worldarena_auto_eval").resolve()
    ensure_dir(work_dir)
    log(f"results_dir: {results_dir}")
    log(f"work_dir: {work_dir}")

    # Prepared directories
    gt_source_dir = work_dir / "gt_source"
    gt_first_frames_dir = work_dir / "gt_first_frames"
    gen_test_dir = work_dir / f"{args.model_name}_test"
    gen_vlm_dir = work_dir / f"{args.model_name}_vlm"
    jepa_gt_dir = work_dir / "jepa_gt"
    jepa_gen_dir = work_dir / "jepa_gen"

    for d in [gt_source_dir, gt_first_frames_dir, gen_test_dir, gen_vlm_dir, jepa_gt_dir, jepa_gen_dir]:
        ensure_dir(d)

    records = load_jsonl(records_path)
    if not records:
        raise RuntimeError(f"No valid records found in {records_path}")
    log(f"Loaded {len(records)} records.")

    used_ids: set[str] = set()
    prepared: List[PreparedSample] = []

    for idx, rec in enumerate(records):
        rec_video = str(rec.get("video_path", ""))
        input_video = resolve_input_video(rec_video, videos_dir)
        if input_video is None:
            log(f"[WARN] Skip record {idx}, video not found: {rec_video}")
            continue

        video_id = make_video_id(rec, input_video.name, idx)
        while video_id in used_ids:
            video_id = f"{video_id}_dup"
        used_ids.add(video_id)

        gt_video = gt_source_dir / args.task_name / "lv1" / "lv2" / "lv3" / f"{video_id}.mp4"
        gen_std_video = gen_test_dir / f"{args.task_name}_{video_id}.mp4"
        first_frame = gt_first_frames_dir / f"{video_id}.png"

        if args.force_rebuild or not (gt_video.exists() and gen_std_video.exists() and first_frame.exists()):
            n_frames = split_stitched_video(
                src_video=input_video,
                gt_video_out=gt_video,
                gen_video_out=gen_std_video,
                first_frame_out=first_frame,
                row_mode=args.row_mode,
            )
            log(f"Prepared {input_video.name} -> frames={n_frames}, id={video_id}")

        gen_vlm_video = gen_vlm_dir / f"{video_id}.mp4"
        gen_jepa_video = jepa_gen_dir / f"{video_id}.mp4"
        gt_jepa_video = jepa_gt_dir / f"{video_id}.mp4"
        copy_if_missing(gen_std_video, gen_vlm_video)
        copy_if_missing(gen_std_video, gen_jepa_video)
        copy_if_missing(gt_video, gt_jepa_video)

        prompt = str(rec.get("prompt", "")).strip()
        prepared.append(
            PreparedSample(
                video_id=video_id,
                gt_video=gt_video,
                gen_standard_video=gen_std_video,
                gen_vlm_video=gen_vlm_video,
                gen_jepa_video=gen_jepa_video,
                gt_jepa_video=gt_jepa_video,
                first_frame=first_frame,
                prompt=prompt,
            )
        )

    if not prepared:
        raise RuntimeError("No samples prepared. Check inputs.")

    summary_data = [
        {
            "gt_path": str(item.gt_video.resolve()),
            "image": str(item.first_frame.resolve()),
            "prompt": [item.prompt],
        }
        for item in prepared
    ]
    summary_json = work_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    log(f"summary.json written: {summary_json}")

    # Build auto config
    if args.base_config and not args.base_config.exists():
        log(f"[WARN] base config not found: {args.base_config}")
    ckpt = load_base_ckpt(args.base_config if args.base_config else None)
    auto_cfg = make_auto_config(work_dir, args.model_name, ckpt)
    auto_cfg_path = work_dir / "config.auto.yaml"
    write_yaml(auto_cfg_path, auto_cfg)
    log(f"auto config written: {auto_cfg_path}")

    selected_metrics = select_metrics(args.metrics, auto_cfg)
    selected_metrics = apply_metric_exclusion(selected_metrics, args.exclude_metrics)
    has_action_following = "action_following" in selected_metrics
    standard_metrics = [m for m in selected_metrics if m != "action_following"]
    log(f"Selected metrics: {selected_metrics if selected_metrics else 'None'}")

    if args.prepare_only:
        log("Prepare-only mode enabled. Skip evaluations.")
        return

    # Standard metrics
    if args.run_standard and standard_metrics:
        run_cmd(
            [
                py,
                str(video_quality_dir / "preprocess_datasets.py"),
                "--summary_json",
                str(summary_json),
                "--gen_video_dir",
                str(gen_test_dir),
                "--output_base",
                str(work_dir / "data"),
            ]
        )
        if args.resize_generated:
            run_cmd([py, str(video_quality_dir / "processing" / "video_resize.py"), "--config_path", str(auto_cfg_path)])
        else:
            log("Skip generated-frame resize (default behavior; use `--resize_generated` to enable).")
        if "trajectory_accuracy" in standard_metrics:
            run_cmd(
                [py, str(video_quality_dir / "processing" / "detection_tracking.py"), "--config_path", str(auto_cfg_path)],
                allow_failure=args.allow_failure,
            )
        run_cmd(
            [py, str(video_quality_dir / "evaluate.py"), "--dimension", *standard_metrics, "--config", str(auto_cfg_path), "--overwrite"],
            allow_failure=args.allow_failure,
        )
    elif args.run_standard:
        log("No standard metric selected, skip standard evaluation.")

    # action_following
    if args.run_action_following and has_action_following:
        duplicate_test_variants(gen_test_dir)
        run_cmd(
            [
                py,
                str(video_quality_dir / "preprocess_datasets_diversity.py"),
                "--summary_json",
                str(summary_json),
                "--gen_video_dir",
                str(gen_test_dir),
                "--output_base",
                str(work_dir / "data_action_following"),
            ]
        )
        run_cmd(
            [py, str(video_quality_dir / "evaluate.py"), "--dimension", "action_following", "--config", str(auto_cfg_path), "--overwrite"],
            allow_failure=args.allow_failure,
        )
    elif args.run_action_following:
        log("Metric `action_following` not selected/available, skip.")

    # VLM
    vlm_model_path = deep_get(auto_cfg, ["ckpt", "vlm_model"])
    if args.run_vlm:
        if is_placeholder(vlm_model_path):
            log("[WARN] Skip VLM because ckpt.vlm_model is missing/placeholder in base config.")
        else:
            run_cmd(
                [
                    py,
                    str(video_quality_dir / "VLM_judge.py"),
                    "--model_name",
                    args.model_name,
                    "--video_dir",
                    str(gen_vlm_dir),
                    "--summary_json",
                    str(summary_json),
                    "--metrics",
                    "all",
                    "--num_frames",
                    str(args.vlm_num_frames),
                    "--output_root",
                    str(work_dir / "output_VLM"),
                    "--tmp_root",
                    str(work_dir / "tmp_VLM"),
                    "--config_path",
                    str(auto_cfg_path),
                ],
                allow_failure=args.allow_failure,
            )

    # JEPA
    if args.run_jepa:
        run_cmd(
            [
                py,
                str(video_quality_dir / "JEDi" / "batch.py"),
                "--real_dir",
                str(jepa_gt_dir),
                "--gen_dir",
                str(jepa_gen_dir),
                "--output_root",
                str(work_dir / "output_JEDi"),
            ],
            cwd=video_quality_dir / "JEDi",
            allow_failure=args.allow_failure,
        )

    # Aggregate
    if not args.skip_aggregate:
        run_cmd(
            [
                py,
                str(video_quality_dir / "csv_results" / "aggregate_results.py"),
                "--base_dir",
                str(work_dir),
                "--model_name",
                args.model_name,
                "--csv_name",
                args.csv_name,
            ],
            allow_failure=args.allow_failure,
        )

        csv_path = work_dir / "csv_results" / args.csv_name
        if csv_path.exists():
            log(f"Done. Aggregated CSV: {csv_path}")
        else:
            log(f"Done, but CSV not found: {csv_path}")
    else:
        log("Done. Skip aggregation by `--skip_aggregate`.")


if __name__ == "__main__":
    main()
