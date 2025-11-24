# 低配ECS部署指南 (2C1G)

## 问题分析

你的ECS配置：
- CPU: 2核 ✓
- 内存: 1GB ⚠️
- 无PyTorch ✗

挑战：
1. 内存紧张（模型训练需要1-1.5GB）
2. 没有PyTorch环境
3. 资源有限，无法同时运行多个进程

---

## 💡 推荐方案：训练-推理分离架构

### 架构图

```
┌──────────────────────────────────┐
│        本地开发机 / 高配云主机     │
│        (用于模型训练)             │
├──────────────────────────────────┤
│ - 训练初始模型                    │
│ - 持续学习更新                    │
│ - 数据清洗与分析                  │
│ - 可用GPU加速                     │
└────────────┬─────────────────────┘
             │ 上传训练好的模型
             │ rank_model.pt (20MB)
             ↓
┌──────────────────────────────────┐
│        ECS 服务器 (2C1G)          │
│        (仅运行推理和交易)         │
├──────────────────────────────────┤
│ - 加载预训练模型                  │
│ - 候选标的打分（推理）            │
│ - 执行交易（开仓/平仓）           │
│ - 记录交易数据                    │
│ - 下载新数据供本地训练            │
└──────────────────────────────────┘
```

---

## 📋 部署步骤

### 步骤1: 本地环境（训练）

**在你的本地Windows机器上**:

```bash
# 1. 确保已安装完整环境
pip install torch pandas numpy

# 2. 训练模型
python -m scripts.modeling.train_ranker --epochs 50

# 3. 验证模型文件
ls models/
# 应该看到:
#   rank_model.pt          (20MB左右)
#   rank_model_meta.json   (5KB)
```

### 步骤2: ECS环境（仅推理）

**选项A: 使用PyTorch（简单但占内存）**

```bash
# 安装最小PyTorch（仅CPU版本）
pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# 或使用国内镜像
pip3 install torch -i https://mirrors.aliyun.com/pypi/simple/ --index-url https://download.pytorch.org/whl/cpu

# 安装依赖
pip3 install pandas numpy
```

**选项B: 使用ONNX（更优，省内存）**

```bash
# 安装ONNX Runtime（比PyTorch小很多）
pip3 install onnxruntime

# 在本地导出ONNX模型
python scripts/export_to_onnx.py

# 上传 rank_model.onnx 到ECS
```

### 步骤3: 上传模型到ECS

```bash
# 方法1: scp上传
scp models/rank_model.pt user@your-ecs-ip:~/Quant4Little/models/
scp models/rank_model_meta.json user@your-ecs-ip:~/Quant4Little/models/

# 方法2: rsync同步
rsync -avz models/ user@your-ecs-ip:~/Quant4Little/models/

# 方法3: Git（推荐，如果模型不太大）
git add models/
git commit -m "Update model"
git push

# 在ECS上
git pull
```

---

## 🚀 ECS上的运行模式

### 模式1: 纯交易（不含模型）

**内存需求: ~150MB**

```bash
# 运行原有的sim_trader.py（不使用模型）
python scripts/sim_trader.py

# 优点：
# - 内存占用小
# - 不需要PyTorch
# - 运行稳定

# 缺点：
# - 没有模型辅助选股
```

### 模式2: 模型辅助交易（轻量级）

**内存需求: ~400-500MB**

```bash
# 使用轻量级推理器
python scripts/lightweight_ranker.py  # 测试

# 集成到交易系统
# 修改 sim_trader.py 使用 LightweightRanker
```

**优化技巧**:

```python
# 在 sim_trader.py 中
from scripts.lightweight_ranker import LightweightRanker

class Trader:
    def __init__(self):
        # 延迟加载模型（需要时才加载）
        self.ranker = None

    def rank_candidates(self, candidates):
        # 只在需要时加载
        if self.ranker is None:
            self.ranker = LightweightRanker()

        # 使用模型打分
        scores = []
        for candidate in candidates:
            features = self.prepare_features(candidate)
            score, cls, probs = self.ranker.predict(*features)
            scores.append(score)

        # 立即释放不用的数据
        del features

        return scores
```

### 模式3: 定时任务模式

**最省资源的方式**:

```bash
# 每天只运行一次模型推理，其余时间运行交易
# crontab -e

# 每天早上8点运行模型打分，保存结果
0 8 * * * cd ~/Quant4Little && python scripts/daily_rank.py > logs/rank.log 2>&1

# 每分钟检查交易信号（不加载模型）
* * * * * cd ~/Quant4Little && python scripts/check_signals.py
```

---

## 💾 内存优化技巧

### 1. 减小PyTorch内存占用

```bash
# 设置环境变量
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 在Python中
import torch
torch.set_num_threads(1)
```

### 2. 使用Swap（临时方案）

```bash
# 增加1GB Swap（应急用）
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 查看Swap
free -h
```

**注意**: Swap会用硬盘空间，速度慢，不建议长期使用

### 3. 监控内存使用

```bash
# 实时监控
watch -n 1 'free -h && ps aux --sort=-%mem | head -10'

# 在Python中监控
import psutil
process = psutil.Process()
print(f"内存使用: {process.memory_info().rss / 1024**2:.1f} MB")
```

---

## 📦 不同方案对比

| 方案 | 内存需求 | CPU需求 | 需要PyTorch | 推荐度 |
|------|---------|---------|------------|--------|
| 纯交易（无模型） | 150MB | 低 | ✗ | ⭐⭐⭐ |
| 轻量级推理 | 400MB | 中 | ✓ | ⭐⭐⭐⭐ |
| 完整系统（含训练） | 1.5GB | 高 | ✓ | ✗ 不适合 |
| 定时任务模式 | 200MB | 低 | ✗ | ⭐⭐⭐⭐⭐ |
| ONNX推理 | 250MB | 低 | ✗ | ⭐⭐⭐⭐⭐ |

---

## 🎯 推荐配置方案

### 方案A: 最简单（适合快速上手）

```
本地: 训练模型
ECS: 运行原有交易策略（不用模型）

优点：
- 稳定可靠
- 内存充足
- 不需要改代码

缺点：
- 没有模型优化
```

### 方案B: 平衡方案（推荐）

```
本地: 训练模型
ECS:
  1. 每天定时运行模型打分（8:00）
  2. 保存打分结果到CSV
  3. 交易程序读取CSV选股
  4. 不常驻加载模型

优点：
- 有模型辅助
- 内存占用小
- 性能好

实现：
```

创建定时打分脚本 `scripts/daily_rank.py`:

```python
#!/usr/bin/env python3
"""
每日定时模型打分
运行后立即退出，不常驻内存
"""
from scripts.lightweight_ranker import LightweightRanker
import pandas as pd
from pathlib import Path
from datetime import datetime

def main():
    # 加载模型
    ranker = LightweightRanker()

    # 读取今日候选
    today = datetime.now().strftime("%Y%m%d")
    candidates = pd.read_csv(f"data/daily_scans/candidates_{today}.csv")

    # 打分
    scores = []
    for _, row in candidates.iterrows():
        # 准备特征
        features = prepare_features(row)  # 你的特征准备函数

        # 预测
        score, cls, probs = ranker.predict(*features)
        scores.append({
            'symbol': row['symbol'],
            'model_score': score,
            'model_class': cls,
            'prob_0': probs[0],
            'prob_1': probs[1],
            'prob_2': probs[2],
        })

    # 保存结果
    results = pd.DataFrame(scores)
    results = results.sort_values('model_score', ascending=False)
    results.to_csv(f"data/ranked/ranked_{today}.csv", index=False)

    print(f"✓ 完成打分: {len(results)} 个标的")
    print(f"  Top 5: {results.head(5)['symbol'].tolist()}")

if __name__ == "__main__":
    main()
```

修改 `sim_trader.py` 读取打分结果:

```python
class Trader:
    def select_positions(self):
        today = datetime.now().strftime("%Y%m%d")
        ranked_file = f"data/ranked/ranked_{today}.csv"

        if Path(ranked_file).exists():
            # 读取模型打分结果
            ranked = pd.read_csv(ranked_file)
            top_symbols = ranked.head(20)['symbol'].tolist()
        else:
            # 降级到传统策略
            top_symbols = self.traditional_selection()

        return top_symbols
```

### 方案C: 终极优化（最省资源）

```
本地: 训练模型 + 导出ONNX
ECS:
  1. 使用ONNX Runtime（比PyTorch小70%）
  2. 定时打分模式
  3. 交易程序读取打分结果

优点：
- 内存占用最小（~250MB）
- 速度快
- 无需PyTorch

步骤：
```

创建ONNX导出脚本:

<function_calls>
<invoke name="Write">
<parameter name="content">#!/usr/bin/env python3
"""
将PyTorch模型导出为ONNX格式
ONNX模型更小、更快、更省内存
"""
import json
from pathlib import Path

import torch
from scripts.modeling.model import TransformerRanker


def export_to_onnx(model_dir: Path = Path("models")):
    """导出模型为ONNX格式"""

    # 加载元数据
    meta_path = model_dir / "rank_model_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # 加载PyTorch模型
    model_path = model_dir / "rank_model.pt"
    model = TransformerRanker(**meta["model_kwargs"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 创建示例输入
    seq_len = meta["seq_len"]
    seq_dim = meta["model_kwargs"]["seq_dim"]
    feature_dim = meta["model_kwargs"]["feature_dim"]

    dummy_sequence = torch.randn(1, seq_len, seq_dim)
    dummy_features = torch.randn(1, feature_dim)

    # 导出ONNX
    onnx_path = model_dir / "rank_model.onnx"
    torch.onnx.export(
        model,
        (dummy_sequence, dummy_features),
        str(onnx_path),
        input_names=["sequence", "features"],
        output_names=["logits"],
        dynamic_axes={
            "sequence": {0: "batch_size"},
            "features": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )

    print(f"✓ ONNX模型已导出: {onnx_path}")

    # 验证ONNX模型
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))

        # 测试推理
        onnx_inputs = {
            "sequence": dummy_sequence.numpy(),
            "features": dummy_features.numpy(),
        }
        onnx_outputs = session.run(None, onnx_inputs)

        # 对比PyTorch输出
        with torch.no_grad():
            torch_outputs = model(dummy_sequence, dummy_features)

        diff = abs(onnx_outputs[0] - torch_outputs.numpy()).max()
        print(f"✓ ONNX验证成功，最大误差: {diff:.6f}")

        # 文件大小对比
        import os
        pt_size = os.path.getsize(model_path) / 1024**2
        onnx_size = os.path.getsize(onnx_path) / 1024**2
        print(f"\n文件大小:")
        print(f"  PyTorch (.pt):  {pt_size:.2f} MB")
        print(f"  ONNX (.onnx):   {onnx_size:.2f} MB")
        print(f"  节省: {(1 - onnx_size/pt_size) * 100:.1f}%")

    except ImportError:
        print("⚠️  未安装onnxruntime，跳过验证")
        print("   安装: pip install onnxruntime")


if __name__ == "__main__":
    export_to_onnx()
