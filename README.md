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
├── model.py                  # 网络架构定义
├── train.py                  # 实验一：同维度交叉拼接
├── train_dim_mismatch.py     # 实验二：异维度拼接 + 修复
├── visualize.py              # 结果可视化
├── requirements.txt          # 依赖
├── README.md                 # 项目说明
├── results/                  # 实验一结果
│   ├── training_log.json
│   ├── run.log
│   ├── model_a.pth
│   ├── model_b.pth
│   └── plots/
└── results_dim_mismatch/     # 实验二结果
    ├── dim_mismatch_log.json
    └── dim_mismatch.log
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 实验一：同维度交叉拼接
python train.py

# 实验二：异维度拼接 + 修复
python train_dim_mismatch.py

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

## 实验一：同维度交叉拼接

### 论证逻辑

```
                Model A (trained)              Model B (trained)
               ┌──────────────┐              ┌──────────────┐
  Input ──►    │  Encoder_A   │              │  Encoder_B   │
               └──────┬───────┘              └──────┬───────┘
                      │ z_A (32-dim)                 │ z_B (32-dim)
               ┌──────┴───────┐              ┌──────┴───────┐
               │  Decoder_A   │              │  Decoder_B   │
               └──────────────┘              └──────────────┘
                      │                          │
                 ~98% acc                    ~98% acc

Cross (A_enc + B_dec):

  Input ──►  Encoder_A  ──► z_A  ──►  Decoder_B  ──►  ???
                                              │
                                     ~5-7% acc (严重退化)

结论：Decoder_B 期望接收的 z_B 与 Encoder_A 产生的 z_A 语义不匹配
→ 模型对齐是必要的
```

### 结果

| 配置 | 准确率 | 准确率下降 |
|------|--------|-----------|
| Model A (原始) | 98.09% | — |
| Model B (原始) | 98.17% | — |
| A_encoder + B_decoder | 5.28% | -92.81% |
| B_encoder + A_decoder | 7.47% | -90.70% |

潜空间分析：cosine similarity = 0.77，L2 distance = 5.43

---

## 实验二：异维度拼接

当两个模型的 latent dimension 不同时（Model A: 32, Model C: 64），直接拼接会报错。

### 直接拼接 → 报错

```
A_encoder(32) + C_decoder(64):
  RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x32 and 64x128)

C_encoder(64) + A_decoder(32):
  RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x64 and 32x128)
```

### 修复方案

| 方法 | 说明 | 准确率 |
|------|------|--------|
| Zero-padding | A_enc(32) → 补零至 64 → C_dec | 24.62% |
| Truncation | C_enc(64) → 截断至 32 → A_dec | 5.74% |

### 结论

1. **维度不匹配直接报错**：矩阵乘法维度冲突，完全无法运行
2. **Zero-padding**：将小维度补零扩大，能运行但语义信息稀释，准确率大幅下降
3. **Truncation**：将大维度截断，丢失信息，准确率崩溃
4. **两种修复都不理想**：即使能跑，性能也远低于原始模型

→ **模型对齐不仅要求语义对齐，还要求维度对齐**
```

## License

MIT
