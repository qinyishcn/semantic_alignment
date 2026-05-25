# Semantic Alignment Verification

验证神经网络模型对齐（Semantic Alignment）的必要性。

## 实验设计

```
Input (784) ──► Encoder ──► Latent (32) ──► Decoder ──► Output (10)
                  │                              │
                  │   Low-dimensional "semantic   │
                  │      interface" layer          │
```

### 核心思路

1. **训练两个独立模型**：相同架构（encoder-decoder），不同随机种子，分别在 MNIST 上训练
2. **评估原始性能**：记录每个模型的分类准确率
3. **交叉拼接测试**：
   - Model A 的 Encoder + Model B 的 Decoder
   - Model B 的 Encoder + Model A 的 Decoder
4. **分析性能差异**：量化模型不匹配带来的准确率损失
5. **潜空间分析**：比较两个模型的 latent space 统计特征

### 假设

如果 encoder 和 decoder 的潜空间没有对齐，那么交叉拼接会导致严重的性能退化。这证明了：

- **模型对齐的必要性**：不同训练过程产生的 encoder/decoder 不能随意互换
- **语义接口**：低维瓶颈层的语义含义由训练过程决定，不同模型学到的表示不同

## 项目结构

```
semantic_alignment/
├── model.py          # 网络架构定义（Encoder, Decoder, AutoencoderClassifier, CrossModel）
├── train.py          # 训练和评估管道
├── visualize.py      # 结果可视化
├── requirements.txt  # 依赖
├── README.md         # 项目说明
└── results/          # 训练结果（运行后生成）
    ├── training_log.json   # 完整训练日志
    ├── run.log             # 控制台输出日志
    ├── model_a.pth         # Model A 权重
    ├── model_b.pth         # Model B 权重
    └── plots/              # 可视化图表
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行实验
python train.py

# 生成可视化图表
python visualize.py
```

## 网络架构

| 层 | 输入维度 | 输出维度 | 说明 |
|---|---|---|---|
| Encoder Layer 1 | 784 | 256 | ReLU + Dropout(0.2) |
| Encoder Layer 2 | 256 | 128 | ReLU + Dropout(0.2) |
| Encoder Layer 3 | 128 | 32 | ReLU (bottleneck) |
| Decoder Layer 1 | 32 | 128 | ReLU + Dropout(0.2) |
| Decoder Layer 2 | 128 | 256 | ReLU + Dropout(0.2) |
| Decoder Layer 3 | 256 | 10 | Output (digits 0-9) |

## 预期结果

- 原始模型 A/B：~97-98% 准确率
- 交叉拼接模型：显著性能退化（预计降至 ~10-30%）
- 两个模型的 latent space 统计特征存在明显差异

## 论证逻辑

```
                Model A (trained)              Model B (trained)
               ┌──────────────┐              ┌──────────────┐
  Input ──►    │  Encoder_A   │              │  Encoder_B   │
               └──────┬───────┘              └──────┬───────┘
                      │ z_A                          │ z_B
               ┌──────┴───────┐              ┌──────┴───────┐
               │  Decoder_A   │              │  Decoder_B   │
               └──────────────┘              └──────────────┘
                      │                          │
                 ~97-98% acc                 ~97-98% acc

Cross (A_enc + B_dec):

  Input ──►  Encoder_A  ──► z_A  ──►  Decoder_B  ──►  ???
                                              │
                                     ~10-30% acc (退化)

结论：Decoder_B 期望接收的 z_B 与 Encoder_A 产生的 z_A 语义不匹配
→ 模型对齐是必要的
```

## License

MIT
