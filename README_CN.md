# Angela's nanoGPT

[English](README.md) | [中文](README_CN.md)

这是一个面向 Apple Silicon、可阅读且可审计的小型 GPT 实现，同时也是一项关于 context-length order 如何影响训练动力学的三轮研究。

## 两个相连的任务

1. **搭建并训练 nanoGPT。** 根目录中的模块实现模型、字符/BPE 数据管线、训练、评估和文本生成。
2. **研究 training dynamics。** 把这个小模型当作实验仪器，研究不同 context-length curriculum 的训练过程。

第二个任务是当前仓库的科学主线。建议从 [训练动力学研究导航](training_dynamics_research/README.md) 开始阅读。

## 三轮研究进程

| 轮次 | 问题 | 方法升级 | 当前允许的结论 |
|---|---|---|---|
| 1. Exploratory pilot | context-length order 是否重要？ | 单 seed schedules 与若干诊断指标 | 发现了巨大的 descending deficit，但测量问题使它只能作为线索。 |
| 2. Paired pilot | 修正测量后，这个 deficit 是否仍存在？ | 5 个 paired seeds、共享初始化与 data manifests、固定 held-out targets | descending terminal deficit 得到复现；curriculum 对 shuffled/fixed-long 没有可检测优势，但 order 与 terminal context 仍混在一起。 |
| 3. Recovery study | 在共同 long-context 条件下，deficit 是否持久？ | 匹配 block permutation、共同 `T=256` recovery、多 horizon process/anchor panels、预注册规则 | **Large deficit reversed; residual sign reversal unresolved.** |

这个过程本身是研究结果的一部分。仓库当前并不声称 curriculum 普遍更好、ordering 永远不重要，或 recency mechanism 已被唯一识别。

## 最终交付物

- [最终 Technical Report](training_dynamics_research/recovery_study/TECHNICAL_REPORT_final.md)：保留研究过程、错误、预注册、正式结果和仍未知的问题。
- [最终论文源文件](training_dynamics_research/paper_icml2026/paper_final.tex)：署名的 ICML 风格 preprint。
- [最终论文 PDF](output/pdf/context_order_recovery_icml2026_final.pdf)。
- [冻结的 Preregistration](training_dynamics_research/recovery_study/PREREGISTRATION.md)。

只有最终交付物使用 `_final` 后缀。运行代码和机器生成结果保持稳定文件名，避免破坏 imports 与复现命令。

## 目录导航

```text
.
├── model.py, train.py, dataset*.py, eval.py, chat.py
│   └── nanoGPT 核心实现
├── Day 1/, Day 2/, Day 3/
│   └── 早期模型搭建里程碑
├── training_dynamics_research/
│   ├── README.md                    # 三轮研究进程与阅读顺序
│   ├── history/                     # 第一、二轮历史材料
│   ├── recovery_study/              # 第三轮代码、预注册、最终报告
│   ├── paper_icml2026/              # 最终论文源文件与 ICML 样式
│   └── research_methodology/        # 可复用的实验与代码审查方法
├── results/
│   ├── ce3_paired/                  # 第二轮结果
│   └── recovery_study/              # 第三轮正式结果与图片
└── output/pdf/
    └── context_order_recovery_icml2026_final.pdf
```

## 快速开始

```bash
pip install torch tiktoken matplotlib scipy numpy

python train.py --tokenizer char --block_size 256
python chat.py --temperature 0.8 --top_k 40
```

## 复现最终 Recovery Study

```bash
python3 -m unittest training_dynamics_research.recovery_study.test_study
python3 -m training_dynamics_research.recovery_study.runner
python3 -m training_dynamics_research.recovery_study.analyze
```

以上命令均应从仓库根目录运行。

正式实验使用 4.8M 参数的 character-level Transformer、Tiny Shakespeare、每次 update 4,096 个 target tokens，以及 24 GB unified memory 的 Apple M3。
