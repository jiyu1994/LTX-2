"""
示例: 两阶段文本转视频 (TI2VidTwoStagesPipeline)

推荐的生产级管线，两阶段生成：
  Stage 1: 半分辨率生成 + CFG 引导
  Stage 2: 2x 上采样 + 蒸馏 LoRA 精修

默认分辨率 1536x1024，30 步去噪
"""

from ltx_client import LTXClient

SERVER_URL = "http://localhost:60317"


def main():
    client = LTXClient(SERVER_URL)

    # 检查服务状态
    health = client.health()
    print(f"服务状态: {health['status']}")
    print(f"GPU: {health['gpu_name']}")

    # 提交生成任务
    task_id = client.text2video(
        prompt=(
            "A serene mountain lake at sunrise, mist rising from the water surface, "
            "golden light filtering through pine trees, a deer drinking at the water's edge, "
            "gentle ripples spreading across the lake, birds flying in the distant sky"
        ),
        pipeline="two_stages",
        seed=42,
        num_frames=121,  # 约 5 秒 @24fps
    )
    print(f"任务已提交: {task_id}")

    # 等待完成并下载
    client.wait_and_download(task_id, "outputs/text2video_two_stages.mp4")


if __name__ == "__main__":
    main()
