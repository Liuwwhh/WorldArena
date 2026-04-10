import glob
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
from PIL import Image
from tqdm import tqdm


def cal_ssim(gt_img, pd_img):
    return structural_similarity(gt_img, pd_img, channel_axis=-1)


def _collect_frame_paths(video_dir):
    patterns = [
        "frame_*.png",
        "frame_*.jpg",
        "frame_*.jpeg",
        "frame_*.PNG",
        "frame_*.JPG",
        "frame_*.JPEG",
    ]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(video_dir, pattern)))
    # de-duplicate while preserving deterministic order
    paths = sorted(set(paths))

    def _frame_key(path):
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        try:
            return int(stem.split("_")[-1])
        except (ValueError, IndexError):
            return stem

    paths.sort(key=_frame_key)
    return paths


def compute_basic_metrics(gt_path, pd_path, metric_names=["psnr", "ssim"]):

    metric_funcs = dict({
        "psnr": peak_signal_noise_ratio,
        "ssim": cal_ssim,
    })
    res = dict()
    for _ in metric_names:
        assert(_ in metric_funcs)
        res.update({_: dict()})
    
    for task_id in sorted(os.listdir(pd_path)):
        task_path = os.path.join(pd_path, task_id)
        
        for _ in metric_names:
            res[_][task_id] = {}

        for episode_id in tqdm(sorted(os.listdir(task_path))):
            if episode_id.endswith(('.png', '.json')): 
                continue
            
            for _ in metric_names:
                res[_][task_id][episode_id] = {}

            gt_video_dir = os.path.join(gt_path, task_id, episode_id, "video")
            gt_image_list = _collect_frame_paths(gt_video_dir)
            n_frames = len(gt_image_list)

            gid_root = os.path.join(task_path, episode_id)
            gid_list = sorted(
                gid for gid in os.listdir(gid_root) if os.path.isdir(os.path.join(gid_root, gid))
            )
            for gid in gid_list:
                pd_video_dir = os.path.join(task_path, episode_id, gid, "video")
                pd_image_list = _collect_frame_paths(pd_video_dir)
                n_pairs = min(len(pd_image_list), n_frames)
                if len(pd_image_list) != n_frames:
                    print(
                        f"[basic_metrics] frame count mismatch: task={task_id}, episode={episode_id}, gid={gid}, "
                        f"pd={len(pd_image_list)}, gt={n_frames}, using {n_pairs} paired frames."
                    )
                if n_pairs == 0:
                    print(
                        f"[basic_metrics] no valid paired frames: task={task_id}, episode={episode_id}, gid={gid}."
                    )
                    for _ in metric_names:
                        res[_][task_id][episode_id].update({gid: 0.0})
                    continue

                cur_metrics = dict()
                for pd_img, gt_img in zip(pd_image_list[:n_pairs], gt_image_list[:n_pairs]):
                    pd_img = np.asanyarray(Image.open(pd_img))
                    gt_img = np.asanyarray(Image.open(gt_img))
                    if pd_img.shape != gt_img.shape:
                        pd_img = cv2.resize(pd_img, dsize=tuple(gt_img.shape[:2][::-1]), interpolation=cv2.INTER_CUBIC)
                    for _ in metric_names:
                        if _ not in cur_metrics:
                            cur_metrics.update({_: 0.0})
                        cur_metrics[_] += metric_funcs[_](gt_img, pd_img)
                for _ in metric_names:
                    res[_][task_id][episode_id].update({gid: cur_metrics[_] / n_pairs})
    
    return res
