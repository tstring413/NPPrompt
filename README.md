# NPPrompt 论文复现报告 (CPU/低资源适配版)

本仓库是对论文 [NPPrompt](https://github.com/XuandongZhao/NPPrompt) 的代码复现。

针对无 GPU 的实验环境，本项目对原代码进行了深度工程适配，成功在 **AMD Ryzen 7 8845HS CPU / 32GB RAM** 环境下跑通了完整的推理流程，并完成了对比实验。

## 📊 核心实验成果 (Key Results)

为了验证论文方法的有效性，我在 **AGNews** 数据集上设计了一组对比实验：对比 **NPPrompt (EPT)** 自动方法与 **人工 (Manual)** 基线方法的性能差异。

**实验设置：**
* **模型：** `distilroberta-base` (RoBERTa 的轻量蒸馏版)
* **硬件：** CPU Only (AMD 8845HS)
* **批次大小 (Batch Size)：** 8
* **模式：** Zero-shot (零样本评估)

| 方法 | Verbalizer 类型 | 准确率 (Accuracy) | 结论 |
| :--- | :--- | :--- | :--- |
| **Manual (基线)** | 人工定义 (固定映射) | **50.82%** | 依赖人工先验，灵活性差，效果较低。 |
| **NPPrompt (EPT)** | **自动搜索 (论文方法)** | **60.89%** | **提升了 +10.07%**。无需人工干预，验证了论文方法的有效性。 |

> **实验结论：** 虽然由于使用了参数量更小的 `distil` 模型（而非论文中的 `roberta-large`），绝对准确率低于论文原值，但实验成功复现了 **“NPPrompt 显著优于人工基线”** 这一核心趋势（提升约 10%），证明了代码逻辑的正确性。

## 🛠️ 工程适配与修改

为了在 **无 GPU** 环境下稳定运行该框架，我进行了以下关键修改：

1.  **模型轻量化：** 将原论文的 `roberta-large` (需 1.5GB+ 显存) 替换为 `distilroberta-base` (~300MB)，大幅降低内存占用和推理延迟，避免 OOM (内存溢出) 问题。
2.  **代码重构：** 移除了代码中硬编码的 CUDA 依赖 (`.cuda()`)，强制模型与数据在 CPU 上运行。
3.  **参数调整：** 将 `batch_size` 调整为 **8**，实现了推理速度与内存占用的平衡。
4.  **工程规范：**
    * 配置 `.gitignore` 排除数据集与模型权重，保持仓库整洁。
    * 提供 `environment.yml` 与 `requirements.txt` 双重环境配置支持。

## 🚀 快速运行指南 (Quick Start)

### 1. 环境安装
推荐使用 Conda 一键配置环境（包含非 Python 依赖）：

```bash
git clone [https://github.com/tstring413/NPPrompt.git](https://github.com/tstring413/NPPrompt.git)
cd NPPrompt

# 方式一：使用 environment.yml (推荐)
conda env create -f environment.yml
conda activate npprompt_env

# 方式二：使用 pip
# pip install -r requirements.txt

# 如果在国内，建议先配置 HuggingFace 镜像
export HF_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com)

# 运行自动化脚本
bash example_run.sh

export HF_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com)

# 注意：这里手动指定了 verbalizer 为 manual
CUDA_VISIBLE_DEVICES="" python emb_prompt.py \
    --model roberta \
    --model_name_or_path distilroberta-base \
    --result_file "results/distilroberta-base/results_agnews_manual.txt" \
    --openprompt_path . \
    --dataset agnews \
    --template_id 0 \
    --seed 144 \
    --verbalizer manual \
    --select 12 \
    --batch_size 8
```