"""
示例: 批量提交多个任务

一次提交多个生成请求，然后并行等待所有结果。
利用服务端队列机制，任务会自动排队依次执行。
"""

import concurrent.futures

from ltx_client import LTXClient

SERVER_URL = "http://localhost:60317"

PROMPTS = [
    {
        "prompt": "A rocket launching from a desert launchpad, massive plumes of fire and smoke, "
        "camera shaking slightly from the vibration, clear blue sky",
        "pipeline": "distilled",
        "output": "outputs/batch_01_rocket.mp4",
    },
    {
        "prompt": "An underwater coral reef teeming with colorful tropical fish, "
        "sunlight rays penetrating through the water surface, gentle current swaying sea plants",
        "pipeline": "distilled",
        "output": "outputs/batch_02_underwater.mp4",
    },
    {
        "prompt": "A snowy village at night, warm light glowing from cottage windows, "
        "snowflakes gently falling, smoke rising from chimneys, a cozy winter atmosphere",
        "pipeline": "distilled",
        "output": "outputs/batch_03_village.mp4",
    },
]


def submit_task(client: LTXClient, task_cfg: dict) -> tuple[str, str]:
    task_id = client.text2video(
        prompt=task_cfg["prompt"],
        pipeline=task_cfg["pipeline"],
    )
    print(f"已提交: {task_id[:8]}... -> {task_cfg['output']}")
    return task_id, task_cfg["output"]


def download_result(client: LTXClient, task_id: str, output: str) -> str:
    client.wait_and_download(task_id, output, poll_interval=10.0, timeout=1200.0)
    return output


def main():
    client = LTXClient(SERVER_URL)

    # 检查服务
    health = client.health()
    print(f"服务状态: {health['status']} | GPU: {health['gpu_name']}")
    print(f"当前队列: {health['queue_length']} 个任务\n")

    # 批量提交
    print("=== 批量提交任务 ===")
    submitted = [submit_task(client, cfg) for cfg in PROMPTS]
    print(f"\n共提交 {len(submitted)} 个任务，开始等待结果...\n")

    # 并行等待（服务端仍是串行执行，但客户端可以并行轮询）
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(submitted)) as executor:
        futures = {
            executor.submit(download_result, client, tid, out): (tid, out)
            for tid, out in submitted
        }
        for future in concurrent.futures.as_completed(futures):
            tid, out = futures[future]
            try:
                path = future.result()
                print(f"✓ 完成: {path}")
            except Exception as e:
                print(f"✗ 失败 [{tid[:8]}]: {e}")

    print(f"\n全部完成！共 {len(submitted)} 个视频")


if __name__ == "__main__":
    main()
