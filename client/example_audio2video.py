"""
示例: 音频驱动视频生成 (A2VidPipelineTwoStage)

以音频文件作为条件，结合文本提示生成与音频匹配的视频。
输出视频将直接合并原始音频。

工作流:
  1. 上传音频 -> 获得 file_id
  2. 提交 a2vid 生成请求
  3. 等待并下载结果

支持格式: wav, mp3, flac, aac 等
"""

from ltx_client import LTXClient

SERVER_URL = "http://localhost:60317"
AUDIO_PATH = "test_assets/music.wav"  # 替换为你的音频路径


def main():
    client = LTXClient(SERVER_URL)

    # --- 方式一: 使用便捷方法 ---
    print("=== 音频驱动视频（便捷方式） ===")
    task_id = client.audio2video(
        prompt=(
            "A dynamic music visualization with abstract colorful shapes pulsing to the rhythm, "
            "neon lights flowing and morphing, particles exploding with each beat"
        ),
        audio_path=AUDIO_PATH,
        seed=42,
        num_frames=121,
    )
    print(f"任务已提交: {task_id}")
    client.wait_and_download(task_id, "outputs/audio2video.mp4")

    # --- 方式二: 手动分步 + 高级参数 ---
    print("\n=== 音频驱动视频（手动分步） ===")

    # 上传音频
    audio_id = client.upload(AUDIO_PATH)
    print(f"音频已上传: {audio_id}")

    # 提交任务，可指定音频截取范围
    task_id = client.generate(
        pipeline="a2vid",
        prompt=(
            "A singer performing on stage with dramatic lighting, "
            "lips synced to the music, spotlight following the performer"
        ),
        audio_file_id=audio_id,
        audio_start_time=0.0,       # 从音频第 0 秒开始
        audio_max_duration=5.0,     # 最多使用 5 秒音频
        seed=88,
        num_frames=121,
    )
    print(f"任务已提交: {task_id}")
    client.wait_and_download(task_id, "outputs/audio2video_stage.mp4")


if __name__ == "__main__":
    main()
