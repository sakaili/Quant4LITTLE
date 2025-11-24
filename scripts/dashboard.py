#!/usr/bin/env python3
"""
策略运行统计仪表板
显示实时统计数据和模型性能
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def show_signal_statistics():
    """显示信号统计"""
    print_header("📊 信号统计")

    signals_file = ROOT / "data" / "paper_trading" / "signals_history.csv"

    if not signals_file.exists():
        print("❌ 无信号历史数据")
        return

    signals = pd.read_csv(signals_file)
    signals['signal_time'] = pd.to_datetime(signals['signal_time'])
    signals['signal_date'] = signals['signal_time'].dt.date

    # 总体统计
    print(f"\n📈 总体数据:")
    print(f"  总信号数: {len(signals)}")
    print(f"  独立标的: {signals['symbol'].nunique()}")
    print(f"  日期范围: {signals['signal_date'].min()} ~ {signals['signal_date'].max()}")

    # 最近统计
    last_7_days = datetime.now().date() - timedelta(days=7)
    recent = signals[signals['signal_date'] >= last_7_days]

    print(f"\n📅 最近7天:")
    print(f"  信号数: {len(recent)}")
    print(f"  日均: {len(recent) / 7:.1f}")

    # 模型分类
    if 'model_class' in signals.columns:
        print(f"\n🤖 模型分类分布:")
        class_counts = signals['model_class'].value_counts()
        for cls, count in class_counts.items():
            pct = count / len(signals) * 100
            print(f"  Class {cls}: {count} ({pct:.1f}%)")

    # 每日统计
    print(f"\n📆 最近5天明细:")
    last_5_days = datetime.now().date() - timedelta(days=5)
    recent_5d = signals[signals['signal_date'] >= last_5_days]

    daily = recent_5d.groupby('signal_date').agg({
        'symbol': 'count',
        'model_score': 'mean' if 'model_score' in signals.columns else 'first'
    }).rename(columns={'symbol': '信号数'})

    if 'model_score' in signals.columns:
        daily = daily.rename(columns={'model_score': '平均分'})
        print(daily.to_string())
    else:
        print(daily[['信号数']].to_string())


def show_training_data():
    """显示训练数据统计"""
    print_header("📚 训练数据")

    backtest_file = ROOT / "data" / "backtest_trades.csv"

    if not backtest_file.exists():
        print("❌ 无回测交易数据")
        return

    trades = pd.read_csv(backtest_file)

    print(f"\n💹 交易记录:")
    print(f"  总交易数: {len(trades)}")

    if 'pnl_pct' in trades.columns:
        print(f"  平均收益: {trades['pnl_pct'].mean():.2%}")
        print(f"  中位收益: {trades['pnl_pct'].median():.2%}")
        print(f"  胜率: {(trades['pnl_pct'] > 0).sum() / len(trades):.1%}")

        # 收益分布
        print(f"\n📊 收益分布:")
        print(f"  >10%: {(trades['pnl_pct'] > 0.10).sum()} ({(trades['pnl_pct'] > 0.10).sum() / len(trades) * 100:.1f}%)")
        print(f"  5%-10%: {((trades['pnl_pct'] > 0.05) & (trades['pnl_pct'] <= 0.10)).sum()}")
        print(f"  0%-5%: {((trades['pnl_pct'] > 0) & (trades['pnl_pct'] <= 0.05)).sum()}")
        print(f"  亏损: {(trades['pnl_pct'] < 0).sum()} ({(trades['pnl_pct'] < 0).sum() / len(trades) * 100:.1f}%)")

    # 按模型分类
    if 'model_class' in trades.columns:
        print(f"\n🎯 按模型分类表现:")
        for cls in sorted(trades['model_class'].unique()):
            cls_trades = trades[trades['model_class'] == cls]
            if 'pnl_pct' in trades.columns:
                print(f"  Class {cls}: {len(cls_trades)} 笔, "
                      f"平均收益 {cls_trades['pnl_pct'].mean():.2%}, "
                      f"胜率 {(cls_trades['pnl_pct'] > 0).sum() / len(cls_trades):.1%}")


def show_model_info():
    """显示模型信息"""
    print_header("🤖 模型信息")

    model_file = ROOT / "models" / "rank_model.pt"
    model_meta_file = ROOT / "models" / "rank_model_meta.json"

    if not model_file.exists():
        print("❌ 模型文件不存在")
        return

    # 模型文件信息
    model_mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
    model_size = model_file.stat().st_size / 1024 / 1024

    print(f"\n📦 模型文件:")
    print(f"  路径: {model_file}")
    print(f"  大小: {model_size:.2f} MB")
    print(f"  训练时间: {model_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  距今: {(datetime.now() - model_mtime).total_seconds() / 3600:.1f} 小时")

    # 模型元数据
    if model_meta_file.exists():
        import json
        with open(model_meta_file, 'r') as f:
            meta = json.load(f)

        print(f"\n⚙️  模型配置:")
        print(f"  训练样本数: {meta.get('train_samples', 'N/A')}")
        print(f"  测试样本数: {meta.get('test_samples', 'N/A')}")
        print(f"  测试准确率: {meta.get('test_accuracy', 0) * 100:.1f}%")
        print(f"  训练轮数: {meta.get('epochs', 'N/A')}")
        print(f"  序列长度: {meta.get('seq_len', 'N/A')}")


def show_hourly_stats():
    """显示每小时统计"""
    print_header("⏰ 每小时运行统计")

    stats_dir = ROOT / "data" / "statistics"
    current_month = datetime.now().strftime('%Y%m')
    stats_file = stats_dir / f"stats_{current_month}.csv"

    if not stats_file.exists():
        print("❌ 无统计数据")
        return

    stats = pd.read_csv(stats_file)
    stats['timestamp'] = pd.to_datetime(stats['timestamp'])

    print(f"\n📊 本月统计 ({current_month}):")
    print(f"  运行次数: {len(stats)}")
    print(f"  最近运行: {stats['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}")

    # 最近24小时
    last_24h = datetime.now() - timedelta(hours=24)
    recent = stats[stats['timestamp'] >= last_24h]

    if len(recent) > 0:
        print(f"\n⏱️  最近24小时:")
        print(f"  运行次数: {len(recent)}")
        print(f"  平均信号数: {recent['recent_7d_count'].mean():.1f}")

    # 显示最近5次运行
    print(f"\n🕐 最近5次运行:")
    recent_5 = stats.tail(5)[['timestamp', 'total_signals', 'unique_symbols', 'recent_7d_count']]
    recent_5['timestamp'] = recent_5['timestamp'].dt.strftime('%m-%d %H:%M')
    recent_5 = recent_5.rename(columns={
        'timestamp': '时间',
        'total_signals': '总信号',
        'unique_symbols': '标的数',
        'recent_7d_count': '近7天'
    })
    print(recent_5.to_string(index=False))


def show_coin_pool():
    """显示币池信息"""
    print_header("💰 币池信息")

    daily_dir = ROOT / "data" / "daily_klines"
    hourly_dir = ROOT / "data" / "hourly_klines"

    daily_count = len(list(daily_dir.glob("*.csv")))
    hourly_count = len(list(hourly_dir.glob("*.csv")))

    print(f"\n📁 数据文件:")
    print(f"  日线文件: {daily_count} 个")
    print(f"  小时线文件: {hourly_count} 个")

    # 随机显示几个币种
    if daily_count > 0:
        files = list(daily_dir.glob("*.csv"))[:10]
        print(f"\n🪙 示例币种（前10个）:")
        for f in files:
            symbol = f.stem.replace('_1d', '')
            print(f"  - {symbol}")


def main():
    """主函数"""
    print("\n" + "🎯" * 40)
    print("  Quant4Little 策略监控仪表板")
    print("  更新时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("🎯" * 40)

    show_signal_statistics()
    show_training_data()
    show_model_info()
    show_hourly_stats()
    show_coin_pool()

    print("\n" + "=" * 80)
    print("  仪表板显示完成")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
