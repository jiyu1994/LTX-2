"""
训练数据预处理工具

功能:
  1. 帧率重采样: 将所有视频统一到 25fps (LTX-2 默认帧率)
  2. 生成元数据: 筛选有效样本，输出 process_videos.py 所需的 JSONL 文件

用法:
  python data_prepare.py                    -- 只生成元数据 (使用原始视频)
  python data_prepare.py --resample         -- 重采样到25fps + 生成元数据
  python data_prepare.py --resample --workers 8
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_info import DATASET_DIR, VIDEOS_DIR, MIN_DURATION, TARGET_FPS, scan_dataset


def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("错误: 未找到 ffmpeg，请先安装 ffmpeg 并加入 PATH")
        print("  Windows: https://www.gyan.dev/ffmpeg/builds/  (下载 essentials)")
        sys.exit(1)


def _resample_one(src: Path, dst: Path, target_fps: int) -> tuple[str, bool, str]:
    """重采样单个视频到目标帧率，返回 (id, success, message)"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-r", str(target_fps),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",
        "-pix_fmt", "yuv420p",
        str(dst),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return (src.stem, False, result.stderr[-200:])
        return (src.stem, True, "")
    except subprocess.TimeoutExpired:
        return (src.stem, False, "超时")
    except Exception as e:
        return (src.stem, False, str(e))


def resample_videos(samples: list[dict], workers: int = 4):
    """
    将所有非 TARGET_FPS 的视频重采样到 TARGET_FPS，保存到 videos_25fps/ 目录。
    已经是 TARGET_FPS 的视频直接复制。
    """
    _check_ffmpeg()

    out_dir = DATASET_DIR / f"videos_{TARGET_FPS}fps"
    out_dir.mkdir(exist_ok=True)

    valid = [s for s in samples if s["duration"] >= MIN_DURATION and s["has_prompt"]]

    to_resample = []
    to_copy = []
    for s in valid:
        src = DATASET_DIR / s["video_path"]
        dst = out_dir / f"{s['id']}.mp4"
        if dst.exists():
            continue
        if s["fps"] == TARGET_FPS:
            to_copy.append((src, dst))
        else:
            to_resample.append((src, dst))

    print("\n" + "=" * 60)
    print(f"帧率重采样到 {TARGET_FPS}fps")
    print("=" * 60)
    print(f"  有效样本 (>={MIN_DURATION}s + 有prompt): {len(valid)}")
    print(f"  需要重采样: {len(to_resample)}")
    print(f"  直接复制:   {len(to_copy)}")
    print(f"  已存在跳过: {len(valid) - len(to_resample) - len(to_copy)}")
    print(f"  输出目录:   {out_dir}")

    for src, dst in to_copy:
        shutil.copy2(src, dst)
    if to_copy:
        print(f"  已复制 {len(to_copy)} 个 {TARGET_FPS}fps 视频")

    if not to_resample:
        print("  无需重采样，全部完成")
    else:
        print(f"  开始重采样 ({workers} 线程)...")
        done, failed = 0, 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_resample_one, src, dst, TARGET_FPS): src.stem
                for src, dst in to_resample
            }
            for future in as_completed(futures):
                sid, ok, msg = future.result()
                if ok:
                    done += 1
                else:
                    failed += 1
                    print(f"    失败: {sid} - {msg}")
                total = done + failed
                if total % 50 == 0 or total == len(to_resample):
                    print(f"    进度: {total}/{len(to_resample)} (失败: {failed})")

        print(f"  重采样完成: 成功 {done}, 失败 {failed}")

    copied_txt = 0
    for s in valid:
        src_txt = VIDEOS_DIR / f"{s['id']}.txt"
        dst_txt = out_dir / f"{s['id']}.txt"
        if src_txt.exists() and not dst_txt.exists():
            shutil.copy2(src_txt, dst_txt)
            copied_txt += 1
    print(f"  已复制 {copied_txt} 个 txt 文件")


def generate_training_metadata(samples: list[dict], use_resampled: bool = False) -> Path:
    """
    筛选 >= MIN_DURATION 且有 prompt 的样本，生成训练管线所需的 JSONL 元数据文件。
    格式: {"caption": "...", "media_path": "videos_25fps/xxx.mp4"}
    """
    resampled_dir = DATASET_DIR / f"videos_{TARGET_FPS}fps"
    if use_resampled and resampled_dir.exists():
        video_subdir = f"videos_{TARGET_FPS}fps"
    else:
        video_subdir = "videos"

    output_path = DATASET_DIR / "train_metadata.jsonl"

    valid = [s for s in samples if s["duration"] >= MIN_DURATION and s["has_prompt"]]
    if use_resampled:
        valid = [s for s in valid if (DATASET_DIR / video_subdir / f"{s['id']}.mp4").exists()]
    valid.sort(key=lambda s: s["id"])

    with open(output_path, "w", encoding="utf-8") as f:
        for s in valid:
            record = {
                "caption": s["prompt"],
                "media_path": f"{video_subdir}/{s['id']}.mp4",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("训练元数据生成")
    print("=" * 60)
    print(f"  筛选条件: 时长 >= {MIN_DURATION}s 且有 prompt")
    print(f"  视频目录: {video_subdir}/")
    print(f"  有效样本: {len(valid)} / {len(samples)}")
    print(f"  保存路径: {output_path}")

    print(f"\n  排除原因统计:")
    no_prompt = sum(1 for s in samples if not s["has_prompt"])
    too_short = sum(1 for s in samples if s["duration"] < MIN_DURATION and s["has_prompt"])
    both = sum(1 for s in samples if s["duration"] < MIN_DURATION and not s["has_prompt"])
    print(f"    无 prompt:             {no_prompt}")
    print(f"    有 prompt 但时长不足:  {too_short}")
    print(f"    两者皆不满足:          {both}")

    print(f"\n  后续步骤: 使用 process_videos.py 预处理")
    print(f"  推荐 resolution_buckets: --resolution-buckets '512x512x129'")
    print(f"  (129帧 / {TARGET_FPS}fps = {129/TARGET_FPS:.2f}s, 帧数满足 frames %% 8 == 1)")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练数据预处理: 帧率重采样 + 元数据生成")
    parser.add_argument("--resample", action="store_true",
                        help=f"将视频重采样到 {TARGET_FPS}fps 并保存到 videos_{TARGET_FPS}fps/")
    parser.add_argument("--workers", type=int, default=4,
                        help="重采样并行线程数 (默认: 4)")
    args = parser.parse_args()

    print("扫描数据集...")
    samples = scan_dataset()
    if not samples:
        print("未找到有效视频，请检查数据集路径")
        raise SystemExit(1)

    if args.resample:
        resample_videos(samples, workers=args.workers)

    use_resampled = args.resample or (DATASET_DIR / f"videos_{TARGET_FPS}fps").exists()
    generate_training_metadata(samples, use_resampled=use_resampled)
