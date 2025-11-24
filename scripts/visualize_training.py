#!/usr/bin/env python3
"""
可视化训练结果
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_training_log(log_path: Path):
    """从训练日志中解析训练和验证指标"""
    epochs = []
    train_losses = []
    val_losses = []
    val_accs = []
    test_acc = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # 匹配格式: Epoch 1: train_loss=1.1695, val_loss=0.9234, val_acc=1.000
            match = re.search(
                r"Epoch (\d+): train_loss=([\d.]+), val_loss=([\d.]+), val_acc=([\d.]+)",
                line,
            )
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))
                val_acc = float(match.group(4))

                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

            # 匹配测试集准确率: Test metrics: {'loss': 1.9018, 'accuracy': 0.2857, ...}
            test_match = re.search(r"Test metrics:.*'accuracy':\s*([\d.]+)", line)
            if test_match:
                test_acc = float(test_match.group(1))

    return epochs, train_losses, val_losses, val_accs, test_acc


def plot_training_curves(epochs, train_losses, val_losses, val_accs, test_acc, output_path: Path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. 损失曲线
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

    # 标注最佳验证损失
    best_val_idx = np.argmin(val_losses)
    best_epoch = epochs[best_val_idx]
    best_val_loss = val_losses[best_val_idx]
    ax1.plot(best_epoch, best_val_loss, "r*", markersize=15, label=f"Best (Epoch {best_epoch})")
    ax1.annotate(
        f"Best: {best_val_loss:.4f}\n(Epoch {best_epoch})",
        xy=(best_epoch, best_val_loss),
        xytext=(best_epoch + 2, best_val_loss + 0.05),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        color="red",
    )

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. 验证准确率曲线
    ax2 = axes[1]
    ax2.plot(epochs, val_accs, "g-", linewidth=2, marker="o", markersize=5, label="Validation Acc")

    # 添加测试集准确率水平线
    if test_acc is not None:
        ax2.axhline(y=test_acc, color="purple", linestyle="--", linewidth=2, label=f"Test Acc = {test_acc:.3f}")
        # 在右侧标注测试准确率
        ax2.text(
            epochs[-1] * 0.98,
            test_acc + 0.03,
            f"Test: {test_acc:.3f}",
            fontsize=10,
            color="purple",
            ha="right",
        )

    # 标注最佳准确率
    best_acc_idx = np.argmax(val_accs)
    best_acc_epoch = epochs[best_acc_idx]
    best_acc = val_accs[best_acc_idx]
    ax2.plot(best_acc_epoch, best_acc, "r*", markersize=15, label=f"Best Val (Epoch {best_acc_epoch})")
    ax2.annotate(
        f"Best: {best_acc:.3f}\n(Epoch {best_acc_epoch})",
        xy=(best_acc_epoch, best_acc),
        xytext=(best_acc_epoch + 2, best_acc - 0.1),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        color="red",
    )

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Validation and Test Accuracy", fontsize=14, fontweight="bold")
    ax2.set_ylim([-0.05, 1.05])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"训练曲线已保存到: {output_path}")

    return best_epoch, best_val_loss, best_acc


def print_summary(epochs, train_losses, val_losses, val_accs):
    """打印训练摘要"""
    best_val_idx = np.argmin(val_losses)
    best_epoch = epochs[best_val_idx]
    best_val_loss = val_losses[best_val_idx]
    best_acc_idx = np.argmax(val_accs)
    best_acc = val_accs[best_acc_idx]

    print("\n" + "=" * 60)
    print("训练摘要")
    print("=" * 60)
    print(f"总Epoch数: {len(epochs)}")
    print(f"\n最佳验证损失:")
    print(f"  Epoch: {best_epoch}")
    print(f"  训练损失: {train_losses[best_val_idx]:.4f}")
    print(f"  验证损失: {best_val_loss:.4f}")
    print(f"  验证准确率: {val_accs[best_val_idx]:.3f}")

    print(f"\n最佳验证准确率:")
    print(f"  Epoch: {epochs[best_acc_idx]}")
    print(f"  准确率: {best_acc:.3f}")

    print(f"\n最终Epoch ({epochs[-1]}):")
    print(f"  训练损失: {train_losses[-1]:.4f}")
    print(f"  验证损失: {val_losses[-1]:.4f}")
    print(f"  验证准确率: {val_accs[-1]:.3f}")

    # 计算过拟合指标
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    overfit_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else 0

    print(f"\n过拟合分析:")
    print(f"  最终训练损失: {final_train_loss:.4f}")
    print(f"  最终验证损失: {final_val_loss:.4f}")
    print(f"  验证/训练比率: {overfit_ratio:.2f}")
    if overfit_ratio > 1.5:
        print("  [!] 模型可能过拟合")
    elif overfit_ratio > 1.2:
        print("  [!] 轻微过拟合")
    else:
        print("  [OK] 拟合良好")

    print("=" * 60 + "\n")


def main():
    log_path = ROOT / "training_50ep.log"

    if not log_path.exists():
        print(f"错误: 找不到训练日志文件 {log_path}")
        print("请先运行训练脚本生成日志")
        return 1

    print(f"正在解析训练日志: {log_path}")
    epochs, train_losses, val_losses, val_accs, test_acc = parse_training_log(log_path)

    if not epochs:
        print("错误: 无法从日志中解析训练数据")
        return 1

    print(f"成功解析 {len(epochs)} 个epoch的数据")
    if test_acc is not None:
        print(f"测试集准确率: {test_acc:.3f}")

    # 打印摘要
    print_summary(epochs, train_losses, val_losses, val_accs)

    # 绘制曲线
    output_path = ROOT / "training_curves.png"
    plot_training_curves(epochs, train_losses, val_losses, val_accs, test_acc, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
