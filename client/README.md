# LTX-2 视频生成服务 - 客户端使用指南

只需要 Python 和 requests 库即可调用服务，不需要 GPU，不需要安装模型。

## 快速开始

1. 安装依赖：pip install requests
2. 把 client 文件夹复制到你的工作目录：cp -r /home/yuji/LTX-2/client ~/my_project/
3. 进入目录：cd ~/my_project/client
4. 运行示例：python example_text2video_fast.py

## 服务地址

- 接口地址：http://localhost:60317
- API 文档：http://localhost:60317/docs （浏览器打开可交互测试）

## 可用管线

| 管线 | 说明 | 速度 | 质量 |
| --- | --- | --- | --- |
| two_stages | 两阶段文转视频（推荐） | 中等 | 最高 |
| two_stages_hq | HQ 版本，res_2s 采样器 | 中等 | 最高 |
| one_stage | 单阶段，低分辨率快速验证 | 快 | 一般 |
| distilled | 蒸馏推理，固定 8 步（最快） | 最快 | 较高 |
| a2vid | 音频驱动视频生成 | 中等 | 较高 |

## 使用示例

### 1. 文本转视频（最简单的用法）

    from ltx_client import LTXClient

    client = LTXClient("http://localhost:60317")
    task_id = client.text2video("A cat playing with a ball of yarn")
    client.wait_and_download(task_id, "output.mp4")

### 2. 文本转视频（带参数）

    from ltx_client import LTXClient

    client = LTXClient("http://localhost:60317")

    task_id = client.text2video(
        prompt="A serene mountain lake at sunrise, mist rising from the water",
        pipeline="two_stages",
        seed=42,
        num_frames=121,
    )
    client.wait_and_download(task_id, "mountain.mp4")

pipeline 可选值：two_stages / two_stages_hq / one_stage / distilled

num_frames 必须满足 8K+1，如 121、193、257

### 3. 图像转视频

    from ltx_client import LTXClient

    client = LTXClient("http://localhost:60317")

    task_id = client.image2video(
        prompt="The scene comes alive with gentle motion, leaves rustling in the wind",
        image_path="input.jpg",
        pipeline="two_stages",
        image_strength=1.0,
    )
    client.wait_and_download(task_id, "image2video.mp4")

image_path 为本地图片路径，客户端会自动上传。image_strength 控制条件强度，范围 0.0 到 1.0。

### 4. 音频驱动视频

    from ltx_client import LTXClient

    client = LTXClient("http://localhost:60317")

    task_id = client.audio2video(
        prompt="A dynamic music visualization with colorful shapes pulsing to the rhythm",
        audio_path="music.wav",
    )
    client.wait_and_download(task_id, "audio2video.mp4")

支持 wav、mp3、flac 等常见音频格式。

### 5. 只提交不等待（异步用法）

    from ltx_client import LTXClient

    client = LTXClient("http://localhost:60317")

    task_id = client.text2video(prompt="A rocket launch", pipeline="distilled")
    print(f"Task ID: {task_id}")

稍后查询状态并下载：

    status = client.task_status(task_id)
    print(status)

    client.download(task_id, "rocket.mp4")

### 6. 纯 curl 调用

提交任务：

    curl -X POST http://localhost:60317/api/generate \
      -H "Content-Type: application/json" \
      -d '{"pipeline":"distilled","prompt":"A cat sitting on a windowsill"}'

查询状态（把 {task_id} 替换为返回的 ID）：

    curl http://localhost:60317/api/task/{task_id}

下载视频：

    curl -o output.mp4 http://localhost:60317/api/task/{task_id}/download

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| prompt | str | 必填 | 视频内容描述（建议英文，200 词以内） |
| pipeline | str | two_stages | 管线类型，见上表 |
| seed | int | 42 | 随机种子，相同种子 + 相同参数 = 相同结果 |
| height | int | 管线默认 | 视频高度（像素），需被 32 整除 |
| width | int | 管线默认 | 视频宽度（像素），需被 32 整除 |
| num_frames | int | 121 | 帧数，必须满足 8K+1 |
| frame_rate | float | 24.0 | 帧率（fps） |
| num_inference_steps | int | 管线默认 | 去噪步数（distilled 管线无需设置） |
| negative_prompt | str | 内置默认 | 不希望出现的内容描述 |
| enhance_prompt | bool | false | 是否自动增强提示词 |
| image_file_id | str | 无 | 图片 file_id（通过 /api/upload 上传获得） |
| image_strength | float | 1.0 | 图片条件强度 |
| audio_file_id | str | 无 | 音频 file_id（仅 a2vid 管线） |

## API 端点一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | /api/health | 服务状态（GPU 信息、队列长度） |
| GET | /api/pipelines | 查看可用管线及默认参数 |
| POST | /api/upload | 上传文件，返回 file_id |
| POST | /api/generate | 提交生成任务，返回 task_id |
| GET | /api/task/{task_id} | 查询任务状态 |
| GET | /api/task/{task_id}/download | 下载生成的视频 |
| GET | /api/queue | 查看队列中的任务 |
| DELETE | /api/task/{task_id} | 删除任务及结果 |

## 提示词写作技巧

- 使用英文撰写提示词，效果最佳
- 按照时间顺序描述画面，像导演描述分镜
- 包含具体细节：动作、外观、镜头角度、光线、色彩
- 控制在 200 词以内
- 直接从主要动作开始，不要写 "Generate a video of..."

示例提示词：

A woman walks through a sunlit garden, her white dress flowing in the gentle
breeze. She reaches out to touch a blooming rose, petals falling softly.
The camera follows her hand in a close-up shot, warm golden hour lighting
casting long shadows across the stone path. Birds chirp in the background.

## 常见问题

### 任务超时了怎么办？

客户端默认等待 10 分钟。大帧数任务需要更长时间，手动指定超时：

    client.wait_and_download(task_id, "output.mp4", timeout=3600.0)

即使客户端超时，服务端任务仍在运行。之后可以手动查询和下载：

    status = client.task_status(task_id)
    client.download(task_id, "output.mp4")

### 提交了多个任务会怎样？

服务端自动排队，同一时间只有一个任务在 GPU 上运行。可通过以下方式查看排队情况：

    client.queue_status()

### 帧数怎么选？

num_frames 必须满足 8K+1（K 为非负整数）：

| 帧数 | 时长（@24fps） | 说明 |
| --- | --- | --- |
| 121 | 约 5 秒 | 默认值，推荐 |
| 193 | 约 8 秒 | 适中 |
| 257 | 约 10 秒 | 较长 |
| 481 | 约 20 秒 | 需要较长生成时间 |

帧数越大，显存占用越多，生成时间越长。建议先用 121 帧测试效果。

### 多个用户同时提交会冲突吗？

不会。每个任务有独立的 task_id，服务端自动排队处理，互不影响。

## 示例脚本

| 文件 | 说明 |
| --- | --- |
| example_text2video.py | 两阶段文转视频（推荐） |
| example_text2video_hq.py | HQ 高质量文转视频 |
| example_text2video_fast.py | 蒸馏快速推理 |
| example_text2video_simple.py | 单阶段低分辨率 |
| example_image2video.py | 图像转视频 |
| example_audio2video.py | 音频驱动视频 |
| example_batch.py | 批量提交多任务 |
