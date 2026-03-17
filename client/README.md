# LTX-2 视频生成服务 - 客户端使用指南

## 快速开始

只需要 Python 和 `requests` 库即可调用服务，**不需要 GPU，不需要安装模型**。

```bash
# 任意目录下操作
pip install requests

# 把 client 文件夹复制到你的工作目录（或直接引用路径）
cp -r /home/yuji/LTX-2/client ~/my_project/
cd ~/my_project/client
```

### 三行代码生成视频

```python
from ltx_client import LTXClient

client = LTXClient("http://localhost:60317")
task_id = client.text2video("A cat playing with a ball of yarn")
client.wait_and_download(task_id, "output.mp4")
```

---

## 服务地址

```
http://localhost:60317
```

API 文档（浏览器打开）：http://localhost:60317/docs

---

## 安装

```bash
pip install requests
```

只需要这一个依赖。Python 3.8+ 均可。

---

## 可用管线

| 管线 | 说明 | 速度 | 质量 |
|---|---|---|---|
| `two_stages` | 两阶段文转视频（推荐） | 中 | ★★★★★ |
| `two_stages_hq` | HQ 版本，res_2s 采样器 | 中 | ★★★★★ |
| `one_stage` | 单阶段，低分辨率快速验证 | 快 | ★★★ |
| `distilled` | 蒸馏推理，固定 8 步（最快） | 最快 | ★★★★ |
| `a2vid` | 音频驱动视频生成 | 中 | ★★★★ |

---

## 使用示例

### 1. 文本转视频

```python
from ltx_client import LTXClient

client = LTXClient("http://localhost:60317")

task_id = client.text2video(
    prompt="A serene mountain lake at sunrise, mist rising from the water",
    pipeline="two_stages",   # 可选: two_stages / two_stages_hq / one_stage / distilled
    seed=42,
    num_frames=121,          # 帧数，必须满足 8K+1（如 121, 193, 257）
)
client.wait_and_download(task_id, "mountain.mp4")
```

### 2. 图像转视频

```python
from ltx_client import LTXClient

client = LTXClient("http://localhost:60317")

task_id = client.image2video(
    prompt="The scene comes alive with gentle motion, leaves rustling in the wind",
    image_path="input.jpg",       # 本地图片路径
    pipeline="two_stages",
    image_strength=1.0,            # 条件强度 0.0~1.0
)
client.wait_and_download(task_id, "image2video.mp4")
```

### 3. 音频驱动视频

```python
from ltx_client import LTXClient

client = LTXClient("http://localhost:60317")

task_id = client.audio2video(
    prompt="A dynamic music visualization with colorful shapes pulsing to the rhythm",
    audio_path="music.wav",       # 本地音频路径（wav/mp3/flac）
)
client.wait_and_download(task_id, "audio2video.mp4")
```

### 4. 只提交不等待（异步用法）

```python
from ltx_client import LTXClient

client = LTXClient("http://localhost:60317")

# 提交任务，立即返回
task_id = client.text2video(prompt="A rocket launch", pipeline="distilled")
print(f"Task ID: {task_id}")  # 记录下来，稍后查询

# ... 做其他事情 ...

# 之后查询状态
status = client.task_status(task_id)
print(status)  # {'status': 'completed', ...}

# 下载结果
client.download(task_id, "rocket.mp4")
```

### 5. 纯 curl 调用（不写 Python）

```bash
# 提交任务
curl -X POST http://localhost:60317/api/generate \
  -H "Content-Type: application/json" \
  -d '{"pipeline":"distilled","prompt":"A cat sitting on a windowsill"}'
# 返回: {"task_id":"xxx-xxx-xxx", ...}

# 查询状态
curl http://localhost:60317/api/task/xxx-xxx-xxx

# 下载视频
curl -o output.mp4 http://localhost:60317/api/task/xxx-xxx-xxx/download
```

---

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `prompt` | str | 必填 | 视频内容描述（建议英文，200 词以内） |
| `pipeline` | str | `"two_stages"` | 管线类型，见上表 |
| `seed` | int | `42` | 随机种子，相同种子 + 相同参数 = 相同结果 |
| `height` | int | 管线默认 | 视频高度（像素），需被 32 整除 |
| `width` | int | 管线默认 | 视频宽度（像素），需被 32 整除 |
| `num_frames` | int | `121` | 帧数，必须满足 `8K+1`（如 9, 25, 57, 121, 193, 257） |
| `frame_rate` | float | `24.0` | 帧率（fps） |
| `num_inference_steps` | int | 管线默认 | 去噪步数（distilled 管线无需设置） |
| `negative_prompt` | str | 内置默认 | 不希望出现的内容描述 |
| `enhance_prompt` | bool | `false` | 是否自动增强提示词 |
| `image_file_id` | str | 无 | 图片 file_id（通过 `/api/upload` 上传获得） |
| `image_strength` | float | `1.0` | 图片条件强度 |
| `audio_file_id` | str | 无 | 音频 file_id（仅 `a2vid` 管线） |

---

## API 端点一览

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/health` | 服务状态（GPU 信息、队列长度） |
| `GET` | `/api/pipelines` | 查看可用管线及默认参数 |
| `POST` | `/api/upload` | 上传文件（图片/音频），返回 `file_id` |
| `POST` | `/api/generate` | 提交生成任务，返回 `task_id` |
| `GET` | `/api/task/{task_id}` | 查询任务状态 |
| `GET` | `/api/task/{task_id}/download` | 下载生成的视频 |
| `GET` | `/api/queue` | 查看队列中的任务 |
| `DELETE` | `/api/task/{task_id}` | 删除任务及结果 |

---


## 常见问题

**Q: 任务超时了怎么办？**

客户端默认等待 10 分钟。大帧数任务需要更长时间，手动指定超时：
```python
client.wait_and_download(task_id, "output.mp4", timeout=3600.0)
```
即使客户端超时，服务端任务仍在运行。可用 `client.task_status(task_id)` 查看进度，完成后再 `client.download(task_id, "output.mp4")` 下载。

**Q: 提交了多个任务会怎样？**

服务端自动排队，同一时间只有一个任务在 GPU 上运行。可通过 `/api/queue` 查看排队情况。

**Q: 帧数怎么选？**

`num_frames` 必须满足 `8K+1`：

| 帧数 | 时长（@24fps） |
|---|---|
| 121 | ~5 秒 |
| 193 | ~8 秒 |
| 257 | ~10 秒 |

帧数越大 → 显存越多 → 生成越慢。建议先用 121 帧测试。

**Q: 多个用户同时提交会冲突吗？**

不会。每个任务有独立的 `task_id`，服务端自动排队处理。

---

## 示例脚本

`client/` 目录下提供了完整示例：

| 文件 | 说明 |
|---|---|
| `example_text2video.py` | 两阶段文转视频（推荐） |
| `example_text2video_hq.py` | HQ 高质量文转视频 |
| `example_text2video_fast.py` | 蒸馏快速推理 |
| `example_text2video_simple.py` | 单阶段低分辨率 |
| `example_image2video.py` | 图像转视频 |
| `example_audio2video.py` | 音频驱动视频 |
| `example_batch.py` | 批量提交多任务 |
