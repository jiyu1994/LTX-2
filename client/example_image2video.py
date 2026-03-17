"""
示例: 图像转视频 (Image-to-Video)

以一张图片作为首帧条件，生成视频。
支持所有管线类型（two_stages / two_stages_hq / distilled / one_stage）。

工作流:
  1. 上传图片 -> 获得 file_id
  2. 提交生成请求，指定 image_file_id
  3. 等待并下载结果
"""

from ltx_client import LTXClient

SERVER_URL = "http://localhost:60317"
IMAGE_PATH = "test_assets/input.jpg"  # 替换为你的图片路径


def main():
    client = LTXClient(SERVER_URL)

    # --- 方式一: 使用便捷方法（自动上传 + 提交） ---
    print("=== 图像转视频（便捷方式） ===")
    task_id = client.image2video(
        prompt=(
            "The scene comes alive with gentle motion, leaves rustling in the wind, "
            "soft sunlight shifting through the clouds, creating dancing shadows"
        ),
        image_path=IMAGE_PATH,
        pipeline="two_stages",
        seed=42,
        image_strength=1.0,  # 图片条件强度 (0.0~1.0)
    )
    print(f"任务已提交: {task_id}")
    client.wait_and_download(task_id, "outputs/image2video.mp4")

    # --- 方式二: 手动分步操作（更灵活） ---
    print("\n=== 图像转视频（手动分步） ===")

    # 步骤 1: 上传图片
    file_id = client.upload(IMAGE_PATH)
    print(f"图片已上传: {file_id}")

    # 步骤 2: 提交任务（可自定义更多参数）
    task_id = client.generate(
        pipeline="two_stages_hq",
        prompt="A beautiful garden with flowers swaying gently in the breeze",
        image_file_id=file_id,
        image_frame_idx=0,   # 将图片放在第 0 帧（首帧）
        image_strength=0.9,  # 稍低的强度给模型更多自由度
        seed=100,
        num_frames=121,
    )
    print(f"任务已提交: {task_id}")
    client.wait_and_download(task_id, "outputs/image2video_hq.mp4")


if __name__ == "__main__":
    main()
