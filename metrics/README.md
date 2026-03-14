# IoU 评测说明

这个目录放 `bbox` 类任务的评测文件。

当前提供：

- `metrics/eval_iou.py`：对 `generated_predictions.jsonl` 计算 IoU

## 使用方法

对单个结果文件做评测：

```bash
python metrics/eval_iou.py \
  saves/bbox_docvqa_bbox_out_1000/qwen3-vl-8b-sft-0313/generated_predictions.jsonl \
  --output-dir metrics/results
```

运行后会生成两个文件：

- `metrics/results/generated_predictions_iou_scored.jsonl`
- `metrics/results/generated_predictions_iou_metrics.json`

如果不传 `--output-dir`，结果会默认写到输入文件同目录。

## 输入格式假设

脚本按你现在的 `generated_predictions.jsonl` 来做：

- `label` 是标准答案
- `predict` 是模型输出

标准标签默认是“按页分组”的格式：

```json
[
  [[x1, y1, x2, y2], ...],
  [[x1, y1, x2, y2], ...]
]
```

其中最外层每一项对应一页图。

## 计算逻辑

### 1. 坐标解析

脚本会优先按 JSON 解析，同时兼容几种常见脏输出：

- ```json fenced code block ```
- Python list 字面量
- 坐标被写成字符串，例如 `["218, 620, 883, 740"]`
- 普通文本里夹着 `[x1, y1, x2, y2]`

### 2. 单框 IoU

对两个框按标准公式计算：

```text
IoU = intersection_area / union_area
```

### 3. 单页多框

同一页如果有多个预测框和多个标注框，不是简单按顺序对齐，而是做“一对一最大匹配”：

- 每个预测框最多匹配一个标注框
- 每个标注框最多匹配一个预测框
- 目标是让匹配后的 IoU 总和最大

页级分数定义为：

```text
page_iou = matched_iou_sum / max(预测框数, 标注框数)
```

这样会同时惩罚漏检和多检。

### 4. 多图 / 多页逻辑

这是这次脚本里专门处理的重点。

如果模型输出本身已经是按页分组的：

```json
[
  [[...]],
  [[...]]
]
```

那就直接按页一一对应计算，每一页只和同页标注比较。

如果模型没有按页分组，只输出了一个扁平框列表：

```json
[
  [x1, y1, x2, y2],
  [x1, y1, x2, y2]
]
```

脚本分两种情况处理：

1. 如果标签是多页，且每页刚好一个标注框，并且预测框数量刚好等于页数
   则按页顺序自动提升成：

```json
[
  [[box1]],
  [[box2]]
]
```

也就是默认模型只是漏掉了最外层的“页维度”。

2. 其他情况
   进入 `flat fallback` 模式，把所有页的标注框先压平成一个集合，再和预测框做全局一对一最大匹配。

这个兜底逻辑的含义是：

- 当模型没有可靠输出页结构时，仍然能得到可计算的 IoU
- 但它不再严格约束“必须命中正确页”
- 所以严格性低于“按页分组评测”

### 5. 样本级分数

样本级 IoU 定义为：

```text
sample_iou = 所有有效匹配的 IoU 总和 / max(总预测框数, 总标注框数)
```

如果是按页分组模式，匹配只在页内进行。
如果是 `flat fallback` 模式，匹配在全局进行。

## 输出字段说明

### `*_iou_scored.jsonl`

每条样本会附带：

- `sample_iou`
- `page_iou`
- `predict_parse_mode`
- `grouped_by_page`
- `parsed_predict_pages`
- `parsed_label_pages`
- `error`

### `*_iou_metrics.json`

汇总结果包括：

- `mean_sample_iou`
- `mean_sample_iou_valid_only`
- `mean_page_iou`
- `mean_page_iou_valid_only`
- `sample_iou_thresholds`
- `page_iou_thresholds`
- `grouped_prediction_samples`
- `flat_fallback_samples`
- `predict_parse_modes`
- `parse_failures`

其中阈值统计默认包含：

- `IoU@0.3`
- `IoU@0.5`
- `IoU@0.7`

## 我是怎么写的

实现思路是这几步：

1. 先把 `predict` 和 `label` 都解析成统一结构：`pages -> boxes -> [x1, y1, x2, y2]`
2. 对异常输出做鲁棒解析，避免因为 code fence 或字符串化坐标直接报废
3. 如果预测能看出页结构，就严格按页算
4. 如果预测没有页结构，但多页样本明显只是漏了一层外括号，就按顺序补回页维度
5. 如果仍然无法可靠恢复页结构，就退化到全局匹配，保证结果可算
6. 页内和全局匹配都用“一对一最大 IoU 匹配”，避免简单按顺序比较带来的误差

这套写法的目的不是只追求“能跑”，而是尽量让：

- 单页样本评测稳定
- 多页样本有明确逻辑
- 模型输出格式不规整时也不会整批失效
