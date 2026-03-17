"""
示例: 蒸馏快速推理 (DistilledPipeline)

最快的推理管线：
  Stage 1: 固定 8 步蒸馏去噪
  Stage 2: 固定 4 步上采样精修
  无需 CFG 引导（无 negative_prompt）

适合批量处理或快速原型验证。
"""

from ltx_client import LTXClient

SERVER_URL = "http://localhost:60317"


def main():
    client = LTXClient(SERVER_URL)

    task_id = client.text2video(
        prompt=(
            "A golden retriever running on a sandy beach, waves crashing in the background, "
            "sunny day with blue sky, the dog splashing through shallow water joyfully"
        ),
        pipeline="distilled",
        seed=77,
        num_frames=121,
        # 蒸馏管线步数固定，无需设置 num_inference_steps
    )
    print(f"任务已提交: {task_id}")

    client.wait_and_download(task_id, "outputs/text2video_fast.mp4")


if __name__ == "__main__":
    main()
