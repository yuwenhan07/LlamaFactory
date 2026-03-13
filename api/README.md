# API Testing for BBox-DocVQA

这个目录放的是用 OpenAI-style API 测试 `bbox_docvqa_*` 数据的最小工具。

## 1. 启动 API

先启动 LlamaFactory 的 API 服务。下面是一个本地 Qwen3-VL 的例子：

```bash
API_PORT=8000 llamafactory-cli api examples/inference/qwen3vl.yaml \
  model_name_or_path=/home/models/Qwen3-VL-8B-Instruct \
  infer_backend=vllm \
  vllm_enforce_eager=true
```

如果你想先走更稳的 HF 后端，可以改成：

```bash
API_PORT=8000 llamafactory-cli api examples/inference/qwen3vl.yaml \
  model_name_or_path=/home/models/Qwen3-VL-8B-Instruct \
  infer_backend=huggingface
```

默认地址是 `http://127.0.0.1:8000/v1`。

## 2. 测试 crop/page/document 数据

### crop 数据

```bash
python api/test_bbox_docvqa_api.py \
  --dataset-jsonl data/bbox_docvqa_crop/bbox_docvqa_rel_crop.jsonl \
  --output api/results_crop.jsonl \
  --model test \
  --max-samples 20
```

### page 数据

```bash
python api/test_bbox_docvqa_api.py \
  --dataset-jsonl data/bbox_docvqa_page/bbox_docvqa_rel_page.jsonl \
  --output api/results_page.jsonl \
  --model test \
  --max-samples 20
```

### document 数据

`bbox_docvqa_document` 的单条样本通常会带很多页图像。你之前的 vLLM 报错就和这个有关。

先看模型/API 能不能处理这么多图，建议先抽样并限制图片数测试：

```bash
python api/test_bbox_docvqa_api.py \
  --dataset-jsonl data/bbox_docvqa_document/bbox_docvqa_document.jsonl \
  --output api/results_document.jsonl \
  --model test \
  --max-samples 5 \
  --max-images 4
```

如果你的 API 后端已经支持更多图片，再去掉 `--max-images` 或者调大。

## 3. 输出格式

脚本会把结果写到一个 JSONL 文件，每行大致包含：

- `sample_idx`
- `bbox_docvqa_id`
- `num_images`
- `predict`
- `label`
- `error`

如果 API 调用失败，`error` 会带上异常信息，方便定位是长度、图片数、显存还是别的问题。

## 4. 常见问题

### `At most 4 image(s) may be provided in one prompt`

说明 API 后端的多图上限太小。对 `document` 数据最常见。

处理办法：

- 先用 `--max-images 4` 做功能验证
- 或者把后端的 `limit_mm_per_prompt.image` 调大

### 本地图片路径不能直接发给 API

这个脚本会自动把本地图片转成 `data:image/...;base64,...`，不需要你手工改路径。

### 想只测某几条

可以用：

```bash
python api/test_bbox_docvqa_api.py \
  --dataset-jsonl data/bbox_docvqa_document/bbox_docvqa_document.jsonl \
  --output api/results_debug.jsonl \
  --start-index 0 \
  --max-samples 2 \
  --model test
```
