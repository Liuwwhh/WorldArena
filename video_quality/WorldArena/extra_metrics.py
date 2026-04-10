import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d, interpolate
from tqdm import tqdm

FID_DIMS = 2048
FEATURE_BATCH_SIZE = 64
LPIPS_BATCH_SIZE = 32
FVD_INPUT_RES = 224
FVD_CHUNK_SIZE = 16


def _collect_frame_paths(video_dir: str) -> List[str]:
    patterns = [
        "frame_*.png",
        "frame_*.jpg",
        "frame_*.jpeg",
        "frame_*.PNG",
        "frame_*.JPG",
        "frame_*.JPEG",
    ]
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(str(p) for p in Path(video_dir).glob(pattern))
    paths = sorted(set(paths))

    def _frame_key(path: str):
        stem = Path(path).stem
        try:
            return int(stem.split("_")[-1])
        except (ValueError, IndexError):
            return stem

    paths.sort(key=_frame_key)
    return paths


def _collect_pairs(gt_path: str, pd_path: str) -> List[Dict]:
    pairs: List[Dict] = []
    for task_id in sorted(os.listdir(pd_path)):
        task_path = os.path.join(pd_path, task_id)
        if not os.path.isdir(task_path):
            continue
        for episode_id in sorted(os.listdir(task_path)):
            if episode_id.endswith((".png", ".json")):
                continue
            episode_path = os.path.join(task_path, episode_id)
            if not os.path.isdir(episode_path):
                continue

            gt_video_dir = os.path.join(gt_path, task_id, episode_id, "video")
            gt_frames = _collect_frame_paths(gt_video_dir)
            if not gt_frames:
                continue

            for gid in sorted(os.listdir(episode_path)):
                gid_path = os.path.join(episode_path, gid)
                if not os.path.isdir(gid_path):
                    continue
                pd_video_dir = os.path.join(gid_path, "video")
                pd_frames = _collect_frame_paths(pd_video_dir)
                n = min(len(gt_frames), len(pd_frames))
                if n <= 0:
                    continue
                pairs.append(
                    {
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "gid": gid,
                        "video_path": pd_video_dir,
                        "gt_frames": gt_frames[:n],
                        "pd_frames": pd_frames[:n],
                    }
                )
    return pairs


def _read_rgb_float(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _align_pred_to_gt(gt_img: np.ndarray, pd_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if gt_img.shape == pd_img.shape:
        return gt_img, pd_img
    target_h, target_w = gt_img.shape[:2]
    pd_img = cv2.resize(pd_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    return gt_img, pd_img


def _ensure_torch_hub_checkpoint(local_ckpt: str, filename: str) -> str:
    if not local_ckpt:
        return ""
    src = Path(local_ckpt)
    if not src.is_file():
        raise FileNotFoundError(f"Required checkpoint not found: {local_ckpt}")
    default_hub_dir = Path(torch.hub.get_dir())
    hub_dir = default_hub_dir
    if not default_hub_dir.exists() or not os.access(default_hub_dir, os.W_OK):
        fallback_hub_dir = Path("/tmp/worldarena_torch_hub")
        fallback_hub_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(fallback_hub_dir)
        torch.hub.set_dir(str(fallback_hub_dir))
        hub_dir = fallback_hub_dir
    dst = hub_dir / "checkpoints" / filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return str(dst)


def _compute_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if features.shape[0] <= 0:
        raise RuntimeError("No features to compute statistics.")
    mu = np.mean(features, axis=0)
    if features.shape[0] == 1:
        sigma = np.eye(features.shape[1], dtype=np.float64) * 1e-6
    else:
        sigma = np.cov(features, rowvar=False)
    return mu, sigma


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def _pack_results(entries: List[Dict], overall_override: float = None):
    if overall_override is None:
        if entries:
            overall_override = float(np.mean([float(x["video_results"]) for x in entries]))
        else:
            overall_override = 0.0
    return [float(overall_override), entries]


def compute_mse_metric(gt_path: str, pd_path: str):
    pairs = _collect_pairs(gt_path, pd_path)
    results: List[Dict] = []
    for pair in tqdm(pairs, desc="mse", disable=False):
        mse_sum = 0.0
        n = 0
        for gt_fp, pd_fp in zip(pair["gt_frames"], pair["pd_frames"]):
            gt_img = _read_rgb_float(gt_fp)
            pd_img = _read_rgb_float(pd_fp)
            gt_img, pd_img = _align_pred_to_gt(gt_img, pd_img)
            mse_sum += float(np.mean((pd_img - gt_img) ** 2))
            n += 1
        score = mse_sum / n if n > 0 else 0.0
        results.append({"video_path": pair["video_path"], "video_results": float(score)})
    return _pack_results(results)


def compute_lpips_metric(gt_path: str, pd_path: str, alexnet_ckpt: str = None, device=None):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("`lpips` is required for LPIPS metric. Please install lpips.") from exc

    _ensure_torch_hub_checkpoint(alexnet_ckpt or "", "alexnet-owt-7be5be79.pth")

    pairs = _collect_pairs(gt_path, pd_path)
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex").to(device).eval()
    results: List[Dict] = []

    with torch.no_grad():
        for pair in tqdm(pairs, desc="lpips", disable=False):
            lpips_sum = 0.0
            frame_count = 0
            total = len(pair["gt_frames"])

            for start in range(0, total, LPIPS_BATCH_SIZE):
                gt_batch_np = []
                pd_batch_np = []
                cur_gt = pair["gt_frames"][start:start + LPIPS_BATCH_SIZE]
                cur_pd = pair["pd_frames"][start:start + LPIPS_BATCH_SIZE]

                target_hw = None
                for gt_fp, pd_fp in zip(cur_gt, cur_pd):
                    gt_img = _read_rgb_float(gt_fp)
                    pd_img = _read_rgb_float(pd_fp)
                    gt_img, pd_img = _align_pred_to_gt(gt_img, pd_img)
                    if target_hw is None:
                        target_hw = gt_img.shape[:2]
                    else:
                        h, w = target_hw
                        if gt_img.shape[:2] != target_hw:
                            gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_CUBIC)
                        if pd_img.shape[:2] != target_hw:
                            pd_img = cv2.resize(pd_img, (w, h), interpolation=cv2.INTER_CUBIC)
                    gt_batch_np.append(gt_img)
                    pd_batch_np.append(pd_img)

                if not gt_batch_np:
                    continue

                gt_tensor = torch.from_numpy(np.stack(gt_batch_np)).permute(0, 3, 1, 2).contiguous().float().to(device)
                pd_tensor = torch.from_numpy(np.stack(pd_batch_np)).permute(0, 3, 1, 2).contiguous().float().to(device)
                lpips_batch = model(pd_tensor, gt_tensor, normalize=True)
                lpips_sum += float(lpips_batch.sum().item())
                frame_count += int(lpips_batch.shape[0])

            score = lpips_sum / frame_count if frame_count > 0 else 0.0
            results.append({"video_path": pair["video_path"], "video_results": float(score)})

    return _pack_results(results)


def _extract_inception_features(frames_np: np.ndarray, model, device: torch.device) -> np.ndarray:
    if frames_np.shape[0] <= 0:
        return np.empty((0, FID_DIMS), dtype=np.float32)
    tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous().float()
    acts = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], FEATURE_BATCH_SIZE):
            batch = tensor[start:start + FEATURE_BATCH_SIZE].to(device)
            pred = model(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            acts.append(pred)
    if not acts:
        return np.empty((0, FID_DIMS), dtype=np.float32)
    return np.concatenate(acts, axis=0)


def compute_fid_metric(gt_path: str, pd_path: str, inception_ckpt: str = None, device=None):
    try:
        from pytorch_fid.inception import InceptionV3
    except ImportError as exc:
        raise ImportError("`pytorch-fid` is required for FID metric. Please install pytorch-fid.") from exc

    _ensure_torch_hub_checkpoint(inception_ckpt or "", "pt_inception-2015-12-05-6726825d.pth")

    pairs = _collect_pairs(gt_path, pd_path)
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FID_DIMS]
    model = InceptionV3([block_idx]).to(device).eval()

    global_gt_feats = []
    global_pd_feats = []
    results: List[Dict] = []

    for pair in tqdm(pairs, desc="fid", disable=False):
        pair_gt_all = []
        pair_pd_all = []
        total = len(pair["gt_frames"])
        for start in range(0, total, FEATURE_BATCH_SIZE):
            gt_batch_np = []
            pd_batch_np = []
            cur_gt = pair["gt_frames"][start:start + FEATURE_BATCH_SIZE]
            cur_pd = pair["pd_frames"][start:start + FEATURE_BATCH_SIZE]
            target_hw = None
            for gt_fp, pd_fp in zip(cur_gt, cur_pd):
                gt_img = _read_rgb_float(gt_fp)
                pd_img = _read_rgb_float(pd_fp)
                gt_img, pd_img = _align_pred_to_gt(gt_img, pd_img)
                if target_hw is None:
                    target_hw = gt_img.shape[:2]
                else:
                    h, w = target_hw
                    if gt_img.shape[:2] != target_hw:
                        gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_CUBIC)
                    if pd_img.shape[:2] != target_hw:
                        pd_img = cv2.resize(pd_img, (w, h), interpolation=cv2.INTER_CUBIC)
                gt_batch_np.append(gt_img)
                pd_batch_np.append(pd_img)

            if not gt_batch_np:
                continue

            gt_feats = _extract_inception_features(np.stack(gt_batch_np), model, device)
            pd_feats = _extract_inception_features(np.stack(pd_batch_np), model, device)
            if gt_feats.shape[0] > 0 and pd_feats.shape[0] > 0:
                pair_gt_all.append(gt_feats)
                pair_pd_all.append(pd_feats)

        if not pair_gt_all or not pair_pd_all:
            score = 0.0
        else:
            pair_gt = np.concatenate(pair_gt_all, axis=0)
            pair_pd = np.concatenate(pair_pd_all, axis=0)
            mu1, sigma1 = _compute_stats(pair_gt)
            mu2, sigma2 = _compute_stats(pair_pd)
            score = _frechet_distance(mu1, sigma1, mu2, sigma2)
            global_gt_feats.append(pair_gt)
            global_pd_feats.append(pair_pd)

        results.append({"video_path": pair["video_path"], "video_results": float(score)})

    if global_gt_feats and global_pd_feats:
        gt_all = np.concatenate(global_gt_feats, axis=0)
        pd_all = np.concatenate(global_pd_feats, axis=0)
        mu1, sigma1 = _compute_stats(gt_all)
        mu2, sigma2 = _compute_stats(pd_all)
        overall = _frechet_distance(mu1, sigma1, mu2, sigma2)
    else:
        overall = 0.0

    return _pack_results(results, overall_override=overall)


def _extract_i3d_features(video_np: np.ndarray, i3d_model, device: torch.device, chunk_size: int) -> List[np.ndarray]:
    feats: List[np.ndarray] = []
    n_frames = int(video_np.shape[0])
    if n_frames <= 0:
        return feats

    if n_frames < chunk_size:
        pad_count = chunk_size - n_frames
        pad_frames = np.repeat(video_np[-1:], pad_count, axis=0)
        video_np = np.concatenate([video_np, pad_frames], axis=0)
        n_frames = chunk_size

    for start in range(0, n_frames, chunk_size):
        chunk = video_np[start:start + chunk_size]
        if chunk.shape[0] != chunk_size:
            continue
        tensor = torch.from_numpy(chunk).permute(0, 3, 1, 2).contiguous().float()
        if tensor.shape[-2] != FVD_INPUT_RES or tensor.shape[-1] != FVD_INPUT_RES:
            tensor = interpolate(tensor, size=(FVD_INPUT_RES, FVD_INPUT_RES), mode="bilinear", align_corners=False)
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)
        tensor = 2.0 * tensor - 1.0
        with torch.no_grad():
            try:
                feat = i3d_model(tensor, rescale=False, resize=False, return_features=True)
            except TypeError:
                feat = i3d_model(tensor)
                if isinstance(feat, (tuple, list)):
                    feat = feat[0]
        feats.append(feat.squeeze(0).cpu().numpy())
    return feats


def compute_fvd_metric(gt_path: str, pd_path: str, i3d_ckpt: str = None, device=None, chunk_size: int = FVD_CHUNK_SIZE):
    if not i3d_ckpt:
        raise ValueError("FVD requires `i3d_ckpt` in config.ckpt.fvd.i3d")
    if not os.path.isfile(i3d_ckpt):
        raise FileNotFoundError(f"FVD model not found: {i3d_ckpt}")

    pairs = _collect_pairs(gt_path, pd_path)
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    i3d_model = torch.jit.load(i3d_ckpt, map_location=device).eval()

    global_gt_feats = []
    global_pd_feats = []
    results: List[Dict] = []

    for pair in tqdm(pairs, desc="fvd", disable=False):
        gt_frames = []
        pd_frames = []
        for gt_fp, pd_fp in zip(pair["gt_frames"], pair["pd_frames"]):
            gt_img = _read_rgb_float(gt_fp)
            pd_img = _read_rgb_float(pd_fp)
            gt_img, pd_img = _align_pred_to_gt(gt_img, pd_img)
            gt_frames.append(gt_img)
            pd_frames.append(pd_img)

        if not gt_frames or not pd_frames:
            score = 0.0
            results.append({"video_path": pair["video_path"], "video_results": float(score)})
            continue

        gt_video = np.stack(gt_frames, axis=0)
        pd_video = np.stack(pd_frames, axis=0)

        pair_gt_feats = _extract_i3d_features(gt_video, i3d_model, device, chunk_size=chunk_size)
        pair_pd_feats = _extract_i3d_features(pd_video, i3d_model, device, chunk_size=chunk_size)

        if pair_gt_feats and pair_pd_feats:
            pair_gt = np.asarray(pair_gt_feats, dtype=np.float64)
            pair_pd = np.asarray(pair_pd_feats, dtype=np.float64)
            mu1, sigma1 = _compute_stats(pair_gt)
            mu2, sigma2 = _compute_stats(pair_pd)
            score = _frechet_distance(mu1, sigma1, mu2, sigma2)
            global_gt_feats.extend(pair_gt_feats)
            global_pd_feats.extend(pair_pd_feats)
        else:
            score = 0.0

        results.append({"video_path": pair["video_path"], "video_results": float(score)})

    if global_gt_feats and global_pd_feats:
        gt_all = np.asarray(global_gt_feats, dtype=np.float64)
        pd_all = np.asarray(global_pd_feats, dtype=np.float64)
        mu1, sigma1 = _compute_stats(gt_all)
        mu2, sigma2 = _compute_stats(pd_all)
        overall = _frechet_distance(mu1, sigma1, mu2, sigma2)
    else:
        overall = 0.0

    return _pack_results(results, overall_override=overall)
