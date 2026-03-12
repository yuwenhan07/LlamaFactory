# BBox-DocVQA 批量推理使用说明

本文档说明新增的两个脚本如何配合使用：

- `scripts/bbox_docvqa_to_llamafactory.py`
- `scripts/run_bbox_docvqa_batch.py`

目标是把 `/ywh/BBox-DocVQA/benchmark/bbox-docvqa-rel.jsonl` 转成 LlamaFactory 可直接读取的多模态数据集，然后对多个视觉语言模型做批量推理测试。

## 1. 脚本职责

### 1.1 `bbox_docvqa_to_llamafactory.py`

作用：

- 读取 BBox-DocVQA 原始 `jsonl`
- 根据 `doc_name + category + evidence_page` 找到对应页面 PNG
- 生成 LlamaFactory 标准多模态数据集
- 自动写出 `dataset_info.json`

支持两种图像模式：

- `page`：直接使用整页 PNG
- `crop`：按 `bbox` 把证据区域裁成单独图片，通常更适合这个数据集

### 1.2 `run_bbox_docvqa_batch.py`

作用：

- 读取模型配置 JSON
- 逐个调用 LlamaFactory 推理入口
- 支持 `vllm` 和 `hf` 两种后端
- 为每个模型保存预测结果与指标
- 最后汇总成总表

---

## 2. 第一步：转换数据集

### 2.1 推荐方式：使用 bbox 裁剪图

在仓库根目录运行：

```bash
cd /ywh/LlamaFactory

python scripts/bbox_docvqa_to_llamafactory.py \
  --input-jsonl ../BBox-DocVQA-improve/benchmark/bbox-docvqa-rel.jsonl \
  --benchmark-dir ../BBox-DocVQA-improve/benchmark \
  --dataset-dir data/bbox_docvqa_crop \
  --dataset-name bbox_docvqa_rel_crop \
  --mode crop
```

执行后会生成：

- `data/bbox_docvqa_crop/dataset_info.json`
- `data/bbox_docvqa_crop/bbox_docvqa_rel_crop.jsonl`
- `data/bbox_docvqa_crop/images/`

### 2.2 baseline：使用整页图

如果你想先跑一个整页基线，可以改成：

```bash
python scripts/bbox_docvqa_to_llamafactory.py \
  --input-jsonl ../BBox-DocVQA-improve/benchmark/bbox-docvqa-rel.jsonl \
  --benchmark-dir ../BBox-DocVQA-improve/benchmark \
  --dataset-dir data/bbox_docvqa_page \
  --dataset-name bbox_docvqa_rel_page \
  --mode page
```

### 2.3 常用可选参数

- `--dataset-dir`：输出数据目录，里面会同时放 `dataset_info.json` 和转换后的 `jsonl`
- `--dataset-name`：注册到 `dataset_info.json` 里的名字，后续推理时要用同一个名字
- `--mode`：`page` 或 `crop`
- `--prompt-prefix`：插入到问题前的额外说明
- `--max-samples`：调试时只转换前 N 条样本

例如只转换前 20 条做联调：

```bash
python scripts/bbox_docvqa_to_llamafactory.py \
  --dataset-dir data/bbox_docvqa_debug \
  --dataset-name bbox_docvqa_debug \
  --mode crop \
  --max-samples 20
```

---

## 3. 第二步：准备模型配置

批量脚本读取一个 JSON 文件作为模型清单。仓库里已经提供了示例：

- `examples/inference/bbox_docvqa_models.example.json`

示例内容：

```json
[
  {
    "name": "qwen3-vl-4b",
    "model_name_or_path": "Qwen/Qwen3-VL-4B-Instruct",
    "template": "qwen3_vl_nothink",
    "backend": "vllm",
    "image_max_pixels": 262144,
    "max_new_tokens": 256
  },
  {
    "name": "internvl-4b",
    "model_name_or_path": "OpenGVLab/InternVL3-4B",
    "template": "intern_vl",
    "backend": "hf",
    "precision": "bf16",
    "image_max_pixels": 262144,
    "max_new_tokens": 256
  }
]
```

### 3.1 常见模板对应关系

- `Qwen3-VL`：`qwen3_vl_nothink`
- `InternVL`：`intern_vl`

### 3.2 建议

- `Qwen3-VL` 优先用 `vllm`
- `InternVL` 如果 `vllm` 兼容性不稳定，可以先用 `hf`
- `name` 字段建议写成简短且唯一的名字，后续会直接作为输出目录名

---

## 4. 第三步：运行批量推理

假设你已经生成了裁剪版数据集 `bbox_docvqa_rel_crop`，运行：

```bash
python scripts/run_bbox_docvqa_batch.py \
  --models-json examples/inference/bbox_docvqa_models.example.json \
  --dataset-dir data/bbox_docvqa_crop \
  --dataset-name bbox_docvqa_rel_crop \
  --output-root saves/bbox_docvqa_crop
```

如果你跑的是整页版数据集，就把：

- `--dataset-dir` 改成 `data/bbox_docvqa_page`
- `--dataset-name` 改成 `bbox_docvqa_rel_page`

### 4.1 常用可选参数

- `--backend`：默认后端，模型配置里没写时使用这个值
- `--cutoff-len`：默认上下文长度
- `--max-new-tokens`：默认生成长度
- `--image-max-pixels`：默认图片像素上限
- `--max-samples`：调试时只跑前 N 条
- `--vllm-batch-size`：`vllm` 推理批大小
- `--per-device-eval-batch-size`：`hf` 后端 eval batch size
- `--hf-precision`：`auto`、`bf16`、`fp16`

例如调试前 50 条：

```bash
python scripts/run_bbox_docvqa_batch.py \
  --models-json examples/inference/bbox_docvqa_models.example.json \
  --dataset-dir data/bbox_docvqa_crop \
  --dataset-name bbox_docvqa_rel_crop \
  --output-root saves/bbox_docvqa_debug \
  --max-samples 50
```

---

## 5. 输出结果说明

脚本会在 `--output-root` 下为每个模型创建一个子目录，例如：

```text
saves/bbox_docvqa_crop/
├── qwen3-vl-4b/
├── internvl-4b/
├── summary.json
└── summary.csv
```

每个模型目录下通常包含：

- `generated_predictions.jsonl`
- `scored_predictions.jsonl`
- `metrics.json`

如果后端是 `vllm`，还会额外生成：

- `bleu_rouge.json`

### 5.1 文件含义

`generated_predictions.jsonl`

- 原始预测结果
- 每条通常包含 `prompt`、`predict`、`label`

`scored_predictions.jsonl`

- 在原始结果基础上增加归一化与打分字段
- 包括 `normalized_predict`、`normalized_label`、`exact_match`、`contains_match`、`anls`

`metrics.json`

- 当前模型的平均指标

`summary.json` / `summary.csv`

- 所有模型的总汇总
- 默认按 `anls`、`contains_match`、`exact_match` 排序

---

## 6. 当前指标定义

批量脚本当前内置了三种简化指标：

- `exact_match`：归一化后完全一致
- `contains_match`：预测与答案有包含关系
- `anls`：近似 DocVQA 风格的字符串相似度指标

注意：

- 这里的 `anls` 是脚本内实现的简化版本，用于模型间快速横向比较
- 如果你需要和正式 benchmark 严格对齐，建议后续再接官方评测逻辑

---

## 7. 推荐实践

建议先按下面顺序跑：

1. 先用 `crop` 模式转换数据
2. 先用 `--max-samples 20` 做联调
3. 确认模型模板和后端没问题后，再跑全量
4. 全量结果优先看 `summary.csv`
5. 对表现异常的模型，再打开对应目录里的 `generated_predictions.jsonl` 看具体输出

---

## 8. 一套完整示例

### 8.1 转换数据

```bash
python scripts/bbox_docvqa_to_llamafactory.py \
  --input-jsonl /ywh/BBox-DocVQA/benchmark/bbox-docvqa-rel.jsonl \
  --benchmark-dir /ywh/BBox-DocVQA/benchmark \
  --dataset-dir data/bbox_docvqa_crop \
  --dataset-name bbox_docvqa_rel_crop \
  --mode crop
```

### 8.2 编辑模型列表

修改：

- `examples/inference/bbox_docvqa_models.example.json`

至少确认以下字段：

- `model_name_or_path`
- `template`
- `backend`

### 8.3 批量运行

```bash
python scripts/run_bbox_docvqa_batch.py \
  --models-json examples/inference/bbox_docvqa_models.example.json \
  --dataset-dir data/bbox_docvqa_crop \
  --dataset-name bbox_docvqa_rel_crop \
  --output-root saves/bbox_docvqa_crop
```

---

## 9. 两个脚本的帮助命令

```bash
python scripts/bbox_docvqa_to_llamafactory.py --help
python scripts/run_bbox_docvqa_batch.py --help
```
