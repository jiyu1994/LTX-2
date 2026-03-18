"""
数据集信息统计工具

数据集结构: videos/ 目录下，每个样本由 {id}.mp4 + {id}.txt 组成
用法: python data_info.py
"""

import json
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

DATASET_DIR = Path(r"D:\job\atlascloud\civitaiNSFWVideoDataset_800")
VIDEOS_DIR = DATASET_DIR / "videos"
MIN_DURATION = 5.0
TARGET_FPS = 25


def get_video_info(video_path: str) -> dict | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps > 0 else 0
    return {
        "fps": round(fps, 2),
        "frame_count": int(frame_count),
        "duration": round(duration, 2),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
    }


def scan_dataset() -> list[dict]:
    """扫描数据集，返回每个样本的完整信息"""
    mp4_files = sorted(VIDEOS_DIR.glob("*.mp4"))
    txt_files = {p.stem for p in VIDEOS_DIR.glob("*.txt")}

    samples = []
    for i, mp4 in enumerate(mp4_files):
        sid = mp4.stem
        has_prompt = sid in txt_files
        prompt = ""
        if has_prompt:
            prompt = (VIDEOS_DIR / f"{sid}.txt").read_text(encoding="utf-8", errors="ignore").strip()

        info = get_video_info(str(mp4))
        if info is None:
            print(f"[{i+1}/{len(mp4_files)}] 跳过无法打开的视频: {sid}")
            continue

        samples.append({
            "id": sid,
            "has_prompt": has_prompt,
            "prompt": prompt,
            "prompt_length": len(prompt),
            "video_path": str(mp4.relative_to(DATASET_DIR)),
            **info,
        })

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(mp4_files)}] 已扫描...")

    print(f"扫描完成: {len(samples)}/{len(mp4_files)} 个视频")
    return samples


def print_duration_distribution(samples: list[dict]):
    durations = np.array([s["duration"] for s in samples])

    print("\n" + "=" * 60)
    print("视频时长统计")
    print("=" * 60)
    print(f"  总数:   {len(durations)}")
    print(f"  最短:   {durations.min():.2f}s")
    print(f"  最长:   {durations.max():.2f}s")
    print(f"  均值:   {durations.mean():.2f}s")
    print(f"  中位数: {np.median(durations):.2f}s")
    print(f"  标准差: {durations.std():.2f}s")

    bins = [0, 2, 4, 5, 6, 8, 10, 15, 20, 30, 60, float("inf")]
    labels = [
        " 0-2s", " 2-4s", " 4-5s", " 5-6s", " 6-8s",
        "8-10s", "10-15s", "15-20s", "20-30s", "30-60s", " >60s",
    ]
    counts, _ = np.histogram(durations, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    bar_width = 40

    print(f"\n{'时长区间':>8s}  {'数量':>5s}  {'占比':>6s}  分布")
    print("-" * 60)
    for label, count in zip(labels, counts):
        pct = count / len(durations) * 100
        bar_len = int(count / max_count * bar_width)
        bar = "█" * bar_len
        print(f"{label:>8s}  {count:>5d}  {pct:>5.1f}%  {bar}")

    gte5 = int((durations >= MIN_DURATION).sum())
    lt5 = int((durations < MIN_DURATION).sum())
    print(f"\n  >= {MIN_DURATION}s (可用于训练): {gte5} 个 ({gte5/len(durations)*100:.1f}%)")
    print(f"  <  {MIN_DURATION}s (不足5秒):    {lt5} 个 ({lt5/len(durations)*100:.1f}%)")

    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\n百分位数:")
    for p in percentiles:
        print(f"  P{p:<2d}: {np.percentile(durations, p):.2f}s")


def print_resolution_distribution(samples: list[dict]):
    res_counter = Counter(s["resolution"] for s in samples)
    print("\n" + "=" * 60)
    print("视频分辨率分布 (Top 15)")
    print("=" * 60)
    for res, count in res_counter.most_common(15):
        pct = count / len(samples) * 100
        print(f"  {res:<16s}  {count:>5d}  ({pct:.1f}%)")


def print_fps_distribution(samples: list[dict]):
    fps_counter = Counter(s["fps"] for s in samples)
    print("\n" + "=" * 60)
    print("视频帧率分布")
    print("=" * 60)
    for fps, count in fps_counter.most_common(10):
        pct = count / len(samples) * 100
        print(f"  {fps:>6.1f} fps  {count:>5d}  ({pct:.1f}%)")

    need_resample = sum(1 for s in samples if s["fps"] != TARGET_FPS)
    already_ok = len(samples) - need_resample
    print(f"\n  已经是 {TARGET_FPS}fps: {already_ok} 个")
    print(f"  需要重采样:     {need_resample} 个")


def print_prompt_stats(samples: list[dict]):
    with_prompt = [s for s in samples if s["has_prompt"]]
    without_prompt = [s for s in samples if not s["has_prompt"]]

    print("\n" + "=" * 60)
    print("Prompt 统计")
    print("=" * 60)
    print(f"  有 prompt: {len(with_prompt)}")
    print(f"  无 prompt: {len(without_prompt)}")

    if without_prompt:
        print(f"  缺失 prompt 的视频 ID (前20个):")
        for s in without_prompt[:20]:
            print(f"    - {s['id']}")
        if len(without_prompt) > 20:
            print(f"    ... 还有 {len(without_prompt) - 20} 个")

    if with_prompt:
        lengths = np.array([s["prompt_length"] for s in with_prompt])
        print(f"\n  Prompt 长度:")
        print(f"    最短: {lengths.min()} 字符")
        print(f"    最长: {lengths.max()} 字符")
        print(f"    均值: {lengths.mean():.0f} 字符")
        print(f"    中位: {np.median(lengths):.0f} 字符")


def save_full_report(samples: list[dict], output_path: str | None = None):
    if output_path is None:
        output_path = str(DATASET_DIR / "dataset_info.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"\n完整扫描结果已保存到: {output_path}")


if __name__ == "__main__":
    samples = scan_dataset()
    if not samples:
        print("未找到有效视频，请检查数据集路径")
        raise SystemExit(1)

    print_duration_distribution(samples)
    print_resolution_distribution(samples)
    print_fps_distribution(samples)
    print_prompt_stats(samples)
    save_full_report(samples)
