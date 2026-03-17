"""
LTX-2 Video Generation API Server

启动方式:
    cd LTX-2
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        uvicorn server.app:app --host 0.0.0.0 --port 60317

API 文档: http://localhost:60317/docs
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import (
    DEFAULT_NEGATIVE_PROMPT,
    LTX_2_3_HQ_PARAMS,
    LTX_2_3_PARAMS,
)
from ltx_pipelines.utils.media_io import encode_video

logger = logging.getLogger("ltx2-server")

# =============================================================================
# Configuration
# =============================================================================

CONFIG_PATH = Path(__file__).parent / "config.yaml"
UPLOAD_DIR = Path(__file__).parent / "uploads"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# =============================================================================
# Data Models
# =============================================================================


class TaskStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class PipelineType(str, Enum):
    two_stages = "two_stages"
    two_stages_hq = "two_stages_hq"
    one_stage = "one_stage"
    distilled = "distilled"
    a2vid = "a2vid"


class GenerateRequest(BaseModel):
    pipeline: PipelineType = PipelineType.two_stages
    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    seed: int = 42
    height: int | None = None
    width: int | None = None
    num_frames: int = 121
    frame_rate: float = 24.0
    num_inference_steps: int | None = None
    enhance_prompt: bool = False

    image_file_id: str | None = Field(None, description="通过 /api/upload 上传的图片 file_id")
    image_frame_idx: int = Field(0, description="图片条件帧索引")
    image_strength: float = Field(1.0, description="图片条件强度")

    audio_file_id: str | None = Field(None, description="通过 /api/upload 上传的音频 file_id（仅 a2vid）")
    audio_start_time: float = 0.0
    audio_max_duration: float | None = None


class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    pipeline: str
    prompt: str
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    position_in_queue: int | None = None


class UploadResponse(BaseModel):
    file_id: str
    filename: str


class HealthResponse(BaseModel):
    status: str
    gpu_name: str | None
    gpu_memory_total_gb: float | None
    gpu_memory_used_gb: float | None
    loaded_pipelines: list[str]
    queue_length: int
    active_task: str | None


# =============================================================================
# Task Store
# =============================================================================


class TaskStore:
    """In-memory task state management with TTL-based cleanup."""

    def __init__(self, result_ttl: int = 3600):
        self.tasks: dict[str, dict[str, Any]] = {}
        self.result_ttl = result_ttl

    def create(self, task_id: str, pipeline: str, prompt: str) -> dict:
        task = {
            "task_id": task_id,
            "status": TaskStatus.queued,
            "pipeline": pipeline,
            "prompt": prompt,
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "params": None,
        }
        self.tasks[task_id] = task
        return task

    def get(self, task_id: str) -> dict | None:
        return self.tasks.get(task_id)

    def update(self, task_id: str, **kwargs: Any) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].update(kwargs)

    def cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            tid
            for tid, t in self.tasks.items()
            if t["completed_at"] and (now - t["completed_at"]) > self.result_ttl
        ]
        for tid in expired:
            output_path = OUTPUT_DIR / f"{tid}.mp4"
            if output_path.exists():
                output_path.unlink()
            del self.tasks[tid]


# =============================================================================
# Pipeline Manager
# =============================================================================


class PipelineManager:
    """Lazy pipeline initialization and GPU-serialized inference execution."""

    def __init__(self, config: dict):
        self.config = config
        self.models_cfg = config["models"]
        self.pipelines: dict[str, Any] = {}
        self.enabled = set(config.get("pipelines", []))
        self.gpu_lock = asyncio.Lock()

        quant_str = config.get("quantization")
        if quant_str == "fp8-cast":
            self.quantization = QuantizationPolicy.fp8_cast()
        elif quant_str == "fp8-scaled-mm":
            self.quantization = QuantizationPolicy.fp8_scaled_mm()
        else:
            self.quantization = None

    def _distilled_lora_list(self, strength: float = 0.8) -> list[LoraPathStrengthAndSDOps]:
        return [
            LoraPathStrengthAndSDOps(
                self.models_cfg["distilled_lora"],
                strength,
                LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]

    def _init_pipeline(self, name: str) -> Any:
        logger.info("Initializing pipeline: %s", name)
        from ltx_pipelines.utils.helpers import get_device

        device = get_device()

        if name == "two_stages":
            from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

            return TI2VidTwoStagesPipeline(
                checkpoint_path=self.models_cfg["checkpoint"],
                distilled_lora=self._distilled_lora_list(
                    self.config.get("distilled_lora_strength", 0.8)
                ),
                spatial_upsampler_path=self.models_cfg["spatial_upsampler"],
                gemma_root=self.models_cfg["gemma_root"],
                loras=[],
                device=device,
                quantization=self.quantization,
            )
        elif name == "two_stages_hq":
            from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline

            return TI2VidTwoStagesHQPipeline(
                checkpoint_path=self.models_cfg["checkpoint"],
                distilled_lora=self._distilled_lora_list(),
                distilled_lora_strength_stage_1=self.config.get(
                    "hq_distilled_lora_strength_stage_1", 0.25
                ),
                distilled_lora_strength_stage_2=self.config.get(
                    "hq_distilled_lora_strength_stage_2", 0.5
                ),
                spatial_upsampler_path=self.models_cfg["spatial_upsampler"],
                gemma_root=self.models_cfg["gemma_root"],
                loras=(),
                device=device,
                quantization=self.quantization,
            )
        elif name == "one_stage":
            from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

            return TI2VidOneStagePipeline(
                checkpoint_path=self.models_cfg["checkpoint"],
                gemma_root=self.models_cfg["gemma_root"],
                loras=[],
                device=device,
                quantization=self.quantization,
            )
        elif name == "distilled":
            from ltx_pipelines.distilled import DistilledPipeline

            return DistilledPipeline(
                distilled_checkpoint_path=self.models_cfg["distilled_checkpoint"],
                gemma_root=self.models_cfg["gemma_root"],
                spatial_upsampler_path=self.models_cfg["spatial_upsampler"],
                loras=[],
                device=device,
                quantization=self.quantization,
            )
        elif name == "a2vid":
            from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage

            return A2VidPipelineTwoStage(
                checkpoint_path=self.models_cfg["checkpoint"],
                distilled_lora=self._distilled_lora_list(
                    self.config.get("distilled_lora_strength", 0.8)
                ),
                spatial_upsampler_path=self.models_cfg["spatial_upsampler"],
                gemma_root=self.models_cfg["gemma_root"],
                loras=[],
                device=device,
                quantization=self.quantization,
            )
        else:
            raise ValueError(f"Unknown pipeline: {name}")

    def get_pipeline(self, name: str) -> Any:
        if name not in self.enabled:
            raise ValueError(
                f"Pipeline '{name}' is not enabled. Enabled: {sorted(self.enabled)}"
            )
        if name not in self.pipelines:
            self.pipelines[name] = self._init_pipeline(name)
        return self.pipelines[name]

    @torch.inference_mode()
    def run_inference(self, pipeline_name: str, params: dict) -> str:
        """Run inference synchronously. Returns the output video path."""
        pipeline = self.get_pipeline(pipeline_name)
        task_id = params.pop("task_id")
        output_path = str(OUTPUT_DIR / f"{task_id}.mp4")
        tiling_config = TilingConfig.default()
        num_frames = params.get("num_frames", 121)
        frame_rate = params.get("frame_rate", 24.0)
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        images = params.pop("images", [])
        audio_path = params.pop("audio_path", None)
        audio_start_time = params.pop("audio_start_time", 0.0)
        audio_max_duration = params.pop("audio_max_duration", None)

        if pipeline_name in ("two_stages", "two_stages_hq", "one_stage", "a2vid"):
            video_guider = LTX_2_3_HQ_PARAMS.video_guider_params if pipeline_name == "two_stages_hq" else LTX_2_3_PARAMS.video_guider_params
            audio_guider = LTX_2_3_HQ_PARAMS.audio_guider_params if pipeline_name == "two_stages_hq" else LTX_2_3_PARAMS.audio_guider_params

        if pipeline_name == "two_stages":
            video, audio = pipeline(
                prompt=params["prompt"],
                negative_prompt=params.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
                seed=params.get("seed", 42),
                height=params.get("height", LTX_2_3_PARAMS.stage_2_height),
                width=params.get("width", LTX_2_3_PARAMS.stage_2_width),
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=params.get(
                    "num_inference_steps", LTX_2_3_PARAMS.num_inference_steps
                ),
                video_guider_params=video_guider,
                audio_guider_params=audio_guider,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=params.get("enhance_prompt", False),
            )
        elif pipeline_name == "two_stages_hq":
            video, audio = pipeline(
                prompt=params["prompt"],
                negative_prompt=params.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
                seed=params.get("seed", 42),
                height=params.get("height", LTX_2_3_HQ_PARAMS.stage_2_height),
                width=params.get("width", LTX_2_3_HQ_PARAMS.stage_2_width),
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=params.get(
                    "num_inference_steps", LTX_2_3_HQ_PARAMS.num_inference_steps
                ),
                video_guider_params=video_guider,
                audio_guider_params=audio_guider,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=params.get("enhance_prompt", False),
            )
        elif pipeline_name == "one_stage":
            video, audio = pipeline(
                prompt=params["prompt"],
                negative_prompt=params.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
                seed=params.get("seed", 42),
                height=params.get("height", LTX_2_3_PARAMS.stage_1_height),
                width=params.get("width", LTX_2_3_PARAMS.stage_1_width),
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=params.get(
                    "num_inference_steps", LTX_2_3_PARAMS.num_inference_steps
                ),
                video_guider_params=video_guider,
                audio_guider_params=audio_guider,
                images=images,
                enhance_prompt=params.get("enhance_prompt", False),
            )
        elif pipeline_name == "distilled":
            video, audio = pipeline(
                prompt=params["prompt"],
                seed=params.get("seed", 42),
                height=params.get("height", LTX_2_3_PARAMS.stage_2_height),
                width=params.get("width", LTX_2_3_PARAMS.stage_2_width),
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=params.get("enhance_prompt", False),
            )
        elif pipeline_name == "a2vid":
            if not audio_path:
                raise ValueError("a2vid pipeline requires audio_file_id")
            video, audio = pipeline(
                prompt=params["prompt"],
                negative_prompt=params.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
                seed=params.get("seed", 42),
                height=params.get("height", LTX_2_3_PARAMS.stage_2_height),
                width=params.get("width", LTX_2_3_PARAMS.stage_2_width),
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=params.get(
                    "num_inference_steps", LTX_2_3_PARAMS.num_inference_steps
                ),
                video_guider_params=video_guider,
                images=images,
                audio_path=audio_path,
                audio_start_time=audio_start_time,
                audio_max_duration=audio_max_duration or (num_frames / frame_rate),
                tiling_config=tiling_config,
                enhance_prompt=params.get("enhance_prompt", False),
            )
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )
        return output_path


# =============================================================================
# Background Worker
# =============================================================================

task_queue: asyncio.Queue[str] = asyncio.Queue()
task_store = TaskStore()
pipeline_manager: PipelineManager | None = None
active_task_id: str | None = None


async def worker_loop() -> None:
    """Background worker that processes one inference task at a time."""
    global active_task_id
    while True:
        task_id = await task_queue.get()
        task = task_store.get(task_id)
        if not task:
            task_queue.task_done()
            continue

        active_task_id = task_id
        task_store.update(task_id, status=TaskStatus.processing, started_at=time.time())
        logger.info("Processing task %s (pipeline=%s)", task_id, task["pipeline"])

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                pipeline_manager.run_inference,
                task["pipeline"],
                task["params"],
            )
            task_store.update(
                task_id, status=TaskStatus.completed, completed_at=time.time()
            )
            logger.info("Task %s completed", task_id)
        except Exception:
            logger.exception("Task %s failed", task_id)
            task_store.update(
                task_id,
                status=TaskStatus.failed,
                completed_at=time.time(),
                error=str(Exception),
            )
            import traceback
            task_store.update(task_id, error=traceback.format_exc())
        finally:
            active_task_id = None
            task_queue.task_done()


async def cleanup_loop(config: dict) -> None:
    """Periodically clean up expired results and uploads."""
    upload_ttl = config.get("server", {}).get("upload_ttl", 3600)
    while True:
        await asyncio.sleep(300)
        task_store.cleanup_expired()
        now = time.time()
        if UPLOAD_DIR.exists():
            for f in UPLOAD_DIR.iterdir():
                if f.is_file() and (now - f.stat().st_mtime) > upload_ttl:
                    f.unlink(missing_ok=True)


# =============================================================================
# FastAPI App
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_manager

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = load_config()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    task_store.result_ttl = config.get("server", {}).get("result_ttl", 3600)
    pipeline_manager = PipelineManager(config)

    logger.info("LTX-2 server starting...")
    logger.info("Enabled pipelines: %s", sorted(pipeline_manager.enabled))
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info("GPU Memory: %.1f GB", mem)

    worker_task = asyncio.create_task(worker_loop())
    cleanup_task = asyncio.create_task(cleanup_loop(config))

    yield

    worker_task.cancel()
    cleanup_task.cancel()


app = FastAPI(
    title="LTX-2 Video Generation API",
    description="基于 LTX-2 模型的视频生成服务，支持文本转视频、图像转视频、音频转视频等多种模式。",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/api/health", response_model=HealthResponse, summary="健康检查")
async def health():
    gpu_name = None
    gpu_total = None
    gpu_used = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 2)
        gpu_used = round(torch.cuda.memory_allocated(0) / (1024**3), 2)

    return HealthResponse(
        status="ok",
        gpu_name=gpu_name,
        gpu_memory_total_gb=gpu_total,
        gpu_memory_used_gb=gpu_used,
        loaded_pipelines=sorted(pipeline_manager.pipelines.keys()) if pipeline_manager else [],
        queue_length=task_queue.qsize(),
        active_task=active_task_id,
    )


@app.get("/api/pipelines", summary="可用管线列表")
async def list_pipelines():
    info = {
        "two_stages": {
            "name": "TI2VidTwoStagesPipeline",
            "description": "两阶段文本/图像转视频（推荐，生产级质量）",
            "supports_image": True,
            "supports_audio": False,
            "default_resolution": f"{LTX_2_3_PARAMS.stage_2_width}x{LTX_2_3_PARAMS.stage_2_height}",
            "default_steps": LTX_2_3_PARAMS.num_inference_steps,
        },
        "two_stages_hq": {
            "name": "TI2VidTwoStagesHQPipeline",
            "description": "两阶段 HQ 文本/图像转视频（res_2s 采样器，更高质量）",
            "supports_image": True,
            "supports_audio": False,
            "default_resolution": f"{LTX_2_3_HQ_PARAMS.stage_2_width}x{LTX_2_3_HQ_PARAMS.stage_2_height}",
            "default_steps": LTX_2_3_HQ_PARAMS.num_inference_steps,
        },
        "one_stage": {
            "name": "TI2VidOneStagePipeline",
            "description": "单阶段文本/图像转视频（快速原型验证）",
            "supports_image": True,
            "supports_audio": False,
            "default_resolution": f"{LTX_2_3_PARAMS.stage_1_width}x{LTX_2_3_PARAMS.stage_1_height}",
            "default_steps": LTX_2_3_PARAMS.num_inference_steps,
        },
        "distilled": {
            "name": "DistilledPipeline",
            "description": "蒸馏快速推理（最快，固定 8 步）",
            "supports_image": True,
            "supports_audio": False,
            "default_resolution": f"{LTX_2_3_PARAMS.stage_2_width}x{LTX_2_3_PARAMS.stage_2_height}",
            "default_steps": 8,
        },
        "a2vid": {
            "name": "A2VidPipelineTwoStage",
            "description": "音频驱动视频生成",
            "supports_image": True,
            "supports_audio": True,
            "default_resolution": f"{LTX_2_3_PARAMS.stage_2_width}x{LTX_2_3_PARAMS.stage_2_height}",
            "default_steps": LTX_2_3_PARAMS.num_inference_steps,
        },
    }
    enabled = pipeline_manager.enabled if pipeline_manager else set()
    return {
        "pipelines": {
            k: {**v, "enabled": k in enabled} for k, v in info.items()
        }
    }


@app.post("/api/upload", response_model=UploadResponse, summary="上传文件")
async def upload_file(file: UploadFile = File(...)):
    """上传图片或音频文件，返回 file_id 供生成请求引用。"""
    file_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix if file.filename else ""
    save_path = UPLOAD_DIR / f"{file_id}{suffix}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return UploadResponse(file_id=file_id, filename=file.filename or "unknown")


@app.post("/api/generate", summary="提交视频生成任务")
async def generate(req: GenerateRequest):
    """
    提交一个视频生成任务到队列。返回 task_id，可用于查询状态和下载结果。

    **管线选择:**
    - `two_stages`: 推荐，生产级质量
    - `two_stages_hq`: 更高质量，更少步数
    - `one_stage`: 快速原型
    - `distilled`: 最快推理
    - `a2vid`: 音频驱动（需上传音频）

    **图像转视频:** 先通过 `/api/upload` 上传图片，然后设置 `image_file_id`

    **音频转视频:** 先通过 `/api/upload` 上传音频，然后选择 `a2vid` 管线并设置 `audio_file_id`
    """
    config = load_config()
    max_queue = config.get("server", {}).get("max_queue_size", 20)
    if task_queue.qsize() >= max_queue:
        raise HTTPException(
            status_code=503,
            detail=f"Queue is full ({max_queue} tasks). Please try again later.",
        )

    pipeline_name = req.pipeline.value
    if pipeline_manager and pipeline_name not in pipeline_manager.enabled:
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline '{pipeline_name}' is not enabled. Enabled: {sorted(pipeline_manager.enabled)}",
        )

    images: list[ImageConditioningInput] = []
    if req.image_file_id:
        matches = list(UPLOAD_DIR.glob(f"{req.image_file_id}.*"))
        if not matches:
            raise HTTPException(status_code=404, detail=f"Upload not found: {req.image_file_id}")
        images.append(
            ImageConditioningInput(
                path=str(matches[0]),
                frame_idx=req.image_frame_idx,
                strength=req.image_strength,
            )
        )

    audio_path: str | None = None
    if req.audio_file_id:
        matches = list(UPLOAD_DIR.glob(f"{req.audio_file_id}.*"))
        if not matches:
            raise HTTPException(status_code=404, detail=f"Upload not found: {req.audio_file_id}")
        audio_path = str(matches[0])

    if pipeline_name == "a2vid" and not audio_path:
        raise HTTPException(
            status_code=400,
            detail="a2vid pipeline requires audio_file_id. Upload audio first via /api/upload.",
        )

    task_id = str(uuid.uuid4())
    params = {
        "task_id": task_id,
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "seed": req.seed,
        "num_frames": req.num_frames,
        "frame_rate": req.frame_rate,
        "enhance_prompt": req.enhance_prompt,
        "images": images,
        "audio_path": audio_path,
        "audio_start_time": req.audio_start_time,
        "audio_max_duration": req.audio_max_duration,
    }
    if req.height is not None:
        params["height"] = req.height
    if req.width is not None:
        params["width"] = req.width
    if req.num_inference_steps is not None:
        params["num_inference_steps"] = req.num_inference_steps

    task_store.create(task_id, pipeline_name, req.prompt)
    task_store.update(task_id, params=params)
    await task_queue.put(task_id)

    return {
        "task_id": task_id,
        "status": "queued",
        "position": task_queue.qsize(),
        "message": "Task submitted. Use GET /api/task/{task_id} to check status.",
    }


@app.get("/api/task/{task_id}", response_model=TaskInfo, summary="查询任务状态")
async def get_task(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    position = None
    if task["status"] == TaskStatus.queued:
        position = sum(
            1
            for tid, t in task_store.tasks.items()
            if t["status"] == TaskStatus.queued and t["created_at"] <= task["created_at"]
        )

    return TaskInfo(
        task_id=task["task_id"],
        status=task["status"],
        pipeline=task["pipeline"],
        prompt=task["prompt"],
        created_at=task["created_at"],
        started_at=task["started_at"],
        completed_at=task["completed_at"],
        error=task["error"],
        position_in_queue=position,
    )


@app.get("/api/task/{task_id}/download", summary="下载生成的视频")
async def download_result(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != TaskStatus.completed:
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed yet. Current status: {task['status']}",
        )

    output_path = OUTPUT_DIR / f"{task_id}.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found (may have expired)")

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"ltx2_{task['pipeline']}_{task_id[:8]}.mp4",
    )


@app.get("/api/queue", summary="查看队列状态")
async def queue_status():
    queued = [
        {"task_id": t["task_id"], "pipeline": t["pipeline"], "prompt": t["prompt"][:50]}
        for t in task_store.tasks.values()
        if t["status"] == TaskStatus.queued
    ]
    processing = [
        {"task_id": t["task_id"], "pipeline": t["pipeline"], "prompt": t["prompt"][:50]}
        for t in task_store.tasks.values()
        if t["status"] == TaskStatus.processing
    ]
    return {
        "queue_length": len(queued),
        "processing": processing,
        "queued": queued,
    }


@app.delete("/api/task/{task_id}", summary="取消/删除任务")
async def delete_task(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    output_path = OUTPUT_DIR / f"{task_id}.mp4"
    if output_path.exists():
        output_path.unlink()

    if task["status"] in (TaskStatus.completed, TaskStatus.failed, TaskStatus.queued):
        del task_store.tasks[task_id]
        return {"message": "Task deleted"}
    else:
        return {"message": "Task is currently processing, cannot delete"}
