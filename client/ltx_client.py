"""
LTX-2 API Python 客户端

用法:
    from ltx_client import LTXClient

    client = LTXClient("http://localhost:8000")
    task_id = client.text2video("A cat on the windowsill")
    client.wait_and_download(task_id, "output.mp4")
"""

from __future__ import annotations

import time
from pathlib import Path

import requests


class LTXClient:
    """LTX-2 视频生成 API 客户端"""

    def __init__(self, base_url: str = "http://localhost:60317"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    # ----- 基础方法 -----

    def health(self) -> dict:
        """检查服务健康状态"""
        resp = self.session.get(f"{self.base_url}/api/health")
        resp.raise_for_status()
        return resp.json()

    def pipelines(self) -> dict:
        """获取可用管线列表"""
        resp = self.session.get(f"{self.base_url}/api/pipelines")
        resp.raise_for_status()
        return resp.json()

    def queue_status(self) -> dict:
        """查看任务队列状态"""
        resp = self.session.get(f"{self.base_url}/api/queue")
        resp.raise_for_status()
        return resp.json()

    def upload(self, file_path: str) -> str:
        """上传文件，返回 file_id"""
        path = Path(file_path)
        with open(path, "rb") as f:
            resp = self.session.post(
                f"{self.base_url}/api/upload",
                files={"file": (path.name, f)},
            )
        resp.raise_for_status()
        return resp.json()["file_id"]

    def generate(self, **params) -> str:
        """提交生成任务，返回 task_id"""
        resp = self.session.post(
            f"{self.base_url}/api/generate",
            json=params,
        )
        resp.raise_for_status()
        return resp.json()["task_id"]

    def task_status(self, task_id: str) -> dict:
        """查询任务状态"""
        resp = self.session.get(f"{self.base_url}/api/task/{task_id}")
        resp.raise_for_status()
        return resp.json()

    def download(self, task_id: str, output_path: str) -> str:
        """下载生成的视频"""
        resp = self.session.get(
            f"{self.base_url}/api/task/{task_id}/download",
            stream=True,
        )
        resp.raise_for_status()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return str(out)

    def delete_task(self, task_id: str) -> dict:
        """删除任务及其结果"""
        resp = self.session.delete(f"{self.base_url}/api/task/{task_id}")
        resp.raise_for_status()
        return resp.json()

    # ----- 便捷方法 -----

    def text2video(
        self,
        prompt: str,
        pipeline: str = "two_stages",
        seed: int = 42,
        height: int | None = None,
        width: int | None = None,
        num_frames: int = 121,
        num_inference_steps: int | None = None,
        **kwargs,
    ) -> str:
        """文本转视频，返回 task_id"""
        params = {
            "pipeline": pipeline,
            "prompt": prompt,
            "seed": seed,
            "num_frames": num_frames,
            **kwargs,
        }
        if height is not None:
            params["height"] = height
        if width is not None:
            params["width"] = width
        if num_inference_steps is not None:
            params["num_inference_steps"] = num_inference_steps
        return self.generate(**params)

    def image2video(
        self,
        prompt: str,
        image_path: str,
        pipeline: str = "two_stages",
        image_strength: float = 1.0,
        **kwargs,
    ) -> str:
        """图像转视频，返回 task_id"""
        file_id = self.upload(image_path)
        return self.generate(
            pipeline=pipeline,
            prompt=prompt,
            image_file_id=file_id,
            image_frame_idx=0,
            image_strength=image_strength,
            **kwargs,
        )

    def audio2video(
        self,
        prompt: str,
        audio_path: str,
        **kwargs,
    ) -> str:
        """音频驱动视频生成，返回 task_id"""
        file_id = self.upload(audio_path)
        return self.generate(
            pipeline="a2vid",
            prompt=prompt,
            audio_file_id=file_id,
            **kwargs,
        )

    def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
    ) -> dict:
        """
        轮询等待任务完成。

        Args:
            task_id: 任务 ID
            poll_interval: 轮询间隔（秒）
            timeout: 最大等待时间（秒）

        Returns:
            任务状态 dict

        Raises:
            TimeoutError: 超时
            RuntimeError: 任务失败
        """
        start = time.time()
        while True:
            status = self.task_status(task_id)
            state = status["status"]

            if state == "completed":
                return status
            if state == "failed":
                raise RuntimeError(f"Task failed: {status.get('error', 'unknown')}")

            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Task {task_id} timed out after {timeout:.0f}s (status: {state})"
                )

            pos = status.get("position_in_queue")
            pos_str = f", queue position: {pos}" if pos else ""
            print(
                f"  [{elapsed:.0f}s] Status: {state}{pos_str}",
                flush=True,
            )
            time.sleep(poll_interval)

    def wait_and_download(
        self,
        task_id: str,
        output_path: str,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
    ) -> str:
        """等待任务完成并下载结果视频"""
        print(f"Waiting for task {task_id}...")
        self.wait_for_task(task_id, poll_interval, timeout)
        path = self.download(task_id, output_path)
        print(f"Video saved to: {path}")
        return path
