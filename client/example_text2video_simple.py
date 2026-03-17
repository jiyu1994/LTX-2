"""
示例: 单阶段文本转视频 (TI2VidOneStagePipeline)

单次扩散直接生成目标分辨率视频，无上采样阶段。
速度较快但分辨率较低（默认 768x512）。
适合快速原型验证和学习了解管线工作原理。
"""

from ltx_client import LTXClient

SERVER_URL = "http://localhost:60317"


def main():
    client = LTXClient(SERVER_URL)

    task_id = client.text2video(
        prompt=(
            "A timelapse of clouds moving across a city skyline at dusk, "
            "lights gradually turning on in buildings, warm orange to deep blue gradient sky"
        ),
        pipeline="one_stage",
        seed=99,
        num_frames=121,
        # 单阶段默认分辨率 768x512，可手动指定
        # height=512,
        # width=768,
    )
    print(f"任务已提交: {task_id}")

    client.wait_and_download(task_id, "outputs/text2video_simple.mp4")


if __name__ == "__main__":
    main()
