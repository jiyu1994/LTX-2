"""
示例: 两阶段 HQ 文本转视频 (TI2VidTwoStagesHQPipeline)

使用 res_2s 二阶采样器替代 Euler，更少步数获得更高质量。
默认 15 步即可达到良好效果，分辨率 1920x1088。
"""

from ltx_client import LTXClient

SERVER_URL = "http://localhost:60317"


def main():
    client = LTXClient(SERVER_URL)

    task_id = client.text2video(
        prompt=(
            "A professional chef in a modern kitchen plating an elegant dish, "
            "steam rising from the plate, precise hand movements arranging micro herbs, "
            "warm overhead lighting creating dramatic shadows, shallow depth of field, "
            "cinematic close-up shot"
        ),
        pipeline="two_stages_hq",
        seed=123,
        num_frames=121,
        # HQ 管线默认 15 步，无需手动设置
    )
    print(f"任务已提交: {task_id}")

    client.wait_and_download(task_id, "outputs/text2video_hq.mp4")


if __name__ == "__main__":
    main()
