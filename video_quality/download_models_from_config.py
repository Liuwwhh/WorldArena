#!/usr/bin/env python3
"""
Download WorldArena video_quality model assets referenced by config.yaml comments,
store them under one local folder, and optionally update config.yaml paths.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional
from urllib.request import urlretrieve

import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def log(msg: str) -> None:
    print(f"[download-models] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_url(url: str, dst: Path) -> Path:
    ensure_dir(dst.parent)
    if dst.exists() and dst.stat().st_size > 0:
        log(f"Skip existing: {dst}")
        return dst
    log(f"Downloading URL -> {dst}")
    urlretrieve(url, str(dst))
    return dst


def snapshot(repo_id: str, local_dir: Path) -> Path:
    # Always call snapshot_download to support true resume and fill missing shards.
    # A non-empty folder does not imply a complete snapshot.
    ensure_dir(local_dir)
    log(f"Snapshot sync: {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return local_dir


def find_repo_file(api: HfApi, repo_id: str, candidates: Iterable[str]) -> Optional[str]:
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    lower_map = {f.lower(): f for f in files}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    # fallback: basename match
    for cand in candidates:
        want = Path(cand).name.lower()
        for f in files:
            if Path(f).name.lower() == want:
                return f
    return None


def hf_file(repo_id: str, remote_filename: str, local_dir: Path, local_name: Optional[str] = None) -> Path:
    ensure_dir(local_dir)
    target_name = local_name or Path(remote_filename).name
    target = local_dir / target_name
    if target.exists() and target.stat().st_size > 0:
        log(f"Skip existing: {target}")
        return target
    log(f"HF file download: {repo_id}:{remote_filename} -> {target}")
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=remote_filename,
        repo_type="model",
        resume_download=True,
    )
    shutil.copy2(cached, target)
    return target


def git_clone_shallow(repo_url: str, dst: Path) -> Path:
    if dst.exists() and any(dst.iterdir()):
        log(f"Skip existing git repo: {dst}")
        return dst
    ensure_dir(dst.parent)
    log(f"Git clone: {repo_url} -> {dst}")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dst)],
        check=True,
    )
    return dst


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WorldArena video_quality models.")
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="WorldArena project root.",
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        default=None,
        help="Model target folder. Default: <project_root>/video_quality/models_downloaded",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=None,
        help="config.yaml path. Default: <project_root>/video_quality/config/config.yaml",
    )
    parser.add_argument(
        "--no_update_config",
        action="store_true",
        help="Do not rewrite config.yaml paths after download.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    video_quality_dir = project_root / "video_quality"
    target_dir = (args.target_dir or (video_quality_dir / "models_downloaded")).resolve()
    config_path = (args.config_path or (video_quality_dir / "config" / "config.yaml")).resolve()

    ensure_dir(target_dir)
    log(f"project_root: {project_root}")
    log(f"target_dir: {target_dir}")
    log(f"config_path: {config_path}")

    api = HfApi()

    # 1) action_following
    action_following_pt = download_url(
        "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        target_dir / "action_following" / "ViT-B-32.pt",
    )

    # 2) semantic_alignment
    qwen25_dir = snapshot(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        target_dir / "semantic_alignment" / "Qwen2.5-VL-7B-Instruct",
    )
    clip_patch16_dir = snapshot(
        "openai/clip-vit-base-patch16",
        target_dir / "semantic_alignment" / "clip-vit-base-patch16",
    )

    # 3) depth_accuracy
    depth_anything_dir = snapshot(
        "depth-anything/Depth-Anything-V2-Small-hf",
        target_dir / "depth_accuracy" / "Depth-Anything-V2-Small-hf",
    )

    # 4) aesthetic_quality
    vit_l_14 = hf_file(
        repo_id="jinaai/clip-models",
        remote_filename="ViT-L-14.pt",
        local_dir=target_dir / "aesthetic_quality",
    )
    aesthetic_head = download_url(
        "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true",
        target_dir / "aesthetic_quality" / "sa_0_4_vit_l_14_linear.pth",
    )

    # 5) shared raft-things
    raft_remote = find_repo_file(
        api,
        "RaphaelLiu/EvalCrafter-Models",
        ["RAFT/models/raft-things.pth", "raft-things.pth"],
    )
    if raft_remote is None:
        raise FileNotFoundError("Cannot find raft-things.pth in RaphaelLiu/EvalCrafter-Models")
    raft_things = hf_file(
        repo_id="RaphaelLiu/EvalCrafter-Models",
        remote_filename=raft_remote,
        local_dir=target_dir / "raft",
        local_name="raft-things.pth",
    )

    # 6) photometric_smoothness
    tartan_remote = find_repo_file(
        api,
        "MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
        [
            "Tartan-C-T-TSKH-spring540x960-M.pth",
            "model.pth",
            "model.safetensors",
        ],
    )
    if tartan_remote is None:
        raise FileNotFoundError("Cannot find Tartan model in MemorySlices/Tartan-C-T-TSKH-spring540x960-M")
    tartan_local_name = Path(tartan_remote).name
    tartan_ckpt = hf_file(
        repo_id="MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
        remote_filename=tartan_remote,
        local_dir=target_dir / "photometric_smoothness",
        local_name=tartan_local_name,
    )

    # 7) motion_smoothness
    vfi_remote = find_repo_file(
        api,
        "MCG-NJU/VFIMamba",
        ["model.pkl", "VFIMamba.pkl"],
    )
    if vfi_remote is None:
        raise FileNotFoundError("Cannot find VFIMamba model file in MCG-NJU/VFIMamba")
    vfi_ckpt = hf_file(
        repo_id="MCG-NJU/VFIMamba",
        remote_filename=vfi_remote,
        local_dir=target_dir / "motion_smoothness",
        local_name="VFIMamba.pkl",
    )

    # 8) image_quality
    musiq_remote = find_repo_file(
        api,
        "chaofengc/IQA-PyTorch-Weights",
        ["musiq_spaq_ckpt-358bb6af.pth"],
    )
    if musiq_remote is None:
        raise FileNotFoundError("Cannot find musiq_spaq_ckpt-358bb6af.pth in chaofengc/IQA-PyTorch-Weights")
    musiq_ckpt = hf_file(
        repo_id="chaofengc/IQA-PyTorch-Weights",
        remote_filename=musiq_remote,
        local_dir=target_dir / "image_quality",
        local_name="musiq_spaq_ckpt-358bb6af.pth",
    )

    # 9) subject_consistency
    dino_repo = git_clone_shallow(
        "https://github.com/facebookresearch/dino.git",
        target_dir / "subject_consistency" / "facebookresearch_dino_main",
    )
    dino_weight_remote = find_repo_file(
        api,
        "Xiaomabufei/lumos",
        ["dino_vitbase16_pretrain.pth"],
    )
    if dino_weight_remote is None:
        raise FileNotFoundError("Cannot find dino_vitbase16_pretrain.pth in Xiaomabufei/lumos")
    dino_weight = hf_file(
        repo_id="Xiaomabufei/lumos",
        remote_filename=dino_weight_remote,
        local_dir=target_dir / "subject_consistency",
        local_name="dino_vitbase16_pretrain.pth",
    )

    # 10) SAM3
    sam_dir = target_dir / "sam"
    ensure_dir(sam_dir)
    sam_remote = find_repo_file(api, "facebook/sam3", ["sam3.pt"])
    if sam_remote is None:
        raise FileNotFoundError("Cannot find sam3.pt in facebook/sam3")
    sam3_pt = hf_file(
        repo_id="facebook/sam3",
        remote_filename=sam_remote,
        local_dir=sam_dir,
        local_name="sam3.pt",
    )
    bpe_remote = find_repo_file(api, "OpenGVLab/ViCLIP-B-16-hf", ["bpe_simple_vocab_16e6.txt.gz"])
    if bpe_remote is None:
        raise FileNotFoundError("Cannot find bpe_simple_vocab_16e6.txt.gz in OpenGVLab/ViCLIP-B-16-hf")
    bpe_file = hf_file(
        repo_id="OpenGVLab/ViCLIP-B-16-hf",
        remote_filename=bpe_remote,
        local_dir=sam_dir,
        local_name="bpe_simple_vocab_16e6.txt.gz",
    )

    # 11) VLM model
    qwen3_dir = snapshot(
        "Qwen/Qwen3-VL-8B-Instruct",
        target_dir / "vlm" / "Qwen3-VL-8B-Instruct",
    )

    # 12) JEDi
    jedi_pretrained = video_quality_dir / "JEDi" / "pretrained_models"
    ensure_dir(jedi_pretrained)
    vith16 = download_url(
        "https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar",
        jedi_pretrained / "vith16.pth.tar",
    )
    ssv2_probe = download_url(
        "https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar",
        jedi_pretrained / "ssv2-probe.pth.tar",
    )

    if args.no_update_config:
        log("Skip config update (--no_update_config).")
        return

    cfg = load_yaml(config_path)
    cfg.setdefault("ckpt", {})
    cfg["ckpt"]["action_following"] = str(action_following_pt.resolve())
    cfg["ckpt"].setdefault("semantic_alignment", {})
    cfg["ckpt"]["semantic_alignment"]["caption"] = str(qwen25_dir.resolve())
    cfg["ckpt"]["semantic_alignment"]["CLIP"] = str(clip_patch16_dir.resolve())
    cfg["ckpt"]["depth_accuracy"] = str(depth_anything_dir.resolve())
    cfg["ckpt"].setdefault("aesthetic_quality", {})
    cfg["ckpt"]["aesthetic_quality"]["clip"] = str(vit_l_14.resolve())
    cfg["ckpt"]["aesthetic_quality"]["aesthetic_head"] = str(aesthetic_head.resolve())
    cfg["ckpt"].setdefault("background_consistency", {})
    cfg["ckpt"]["background_consistency"]["clip"] = str(action_following_pt.resolve())
    cfg["ckpt"]["background_consistency"]["raft"] = str(raft_things.resolve())
    cfg["ckpt"].setdefault("dynamic_degree", {})
    cfg["ckpt"]["dynamic_degree"]["raft"] = str(raft_things.resolve())
    cfg["ckpt"].setdefault("flow_score", {})
    cfg["ckpt"]["flow_score"]["raft"] = str(raft_things.resolve())
    cfg["ckpt"].setdefault("photometric_smoothness", {})
    cfg["ckpt"]["photometric_smoothness"]["cfg"] = str(
        (video_quality_dir / "WorldArena" / "third_party" / "SEA-RAFT" / "config" / "eval" / "spring-M.json").resolve()
    )
    cfg["ckpt"]["photometric_smoothness"]["model"] = str(tartan_ckpt.resolve())
    cfg["ckpt"].setdefault("motion_smoothness", {})
    cfg["ckpt"]["motion_smoothness"]["model"] = str(vfi_ckpt.resolve())
    cfg["ckpt"].setdefault("image_quality", {})
    cfg["ckpt"]["image_quality"]["musiq"] = str(musiq_ckpt.resolve())
    cfg["ckpt"].setdefault("subject_consistency", {})
    cfg["ckpt"]["subject_consistency"]["repo"] = str(dino_repo.resolve())
    cfg["ckpt"]["subject_consistency"]["weight"] = str(dino_weight.resolve())
    cfg["ckpt"]["subject_consistency"]["model"] = "dino_vitb16"
    cfg["ckpt"]["subject_consistency"]["raft"] = str(raft_things.resolve())
    cfg["ckpt"]["sam3_model_ckpt"] = str(sam_dir.resolve())
    cfg["ckpt"]["vlm_model"] = str(qwen3_dir.resolve())

    save_yaml(config_path, cfg)
    log(f"Updated config: {config_path}")

    log("Done.")
    log(f"JEDi vith16: {vith16}")
    log(f"JEDi ssv2-probe: {ssv2_probe}")
    log(f"SAM3: {sam3_pt}, {bpe_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        log(f"[ERROR] {exc}")
        sys.exit(1)
