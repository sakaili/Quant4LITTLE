#!/usr/bin/env python3
"""
每小时自动运行的交易脚本（增强版）
1. 更新最新数据
2. 运行Paper Trading生成信号
3. 收集信号数据用于模型训练
4. 定期重新训练模型（每天或累积足够样本）
5. 记录详细统计
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pandas as pd

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 配置日志
log_dir = ROOT / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"hourly_trading_{datetime.now().strftime('%Y%m')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def update_coin_pool():
    """更新币池数据"""
    from scripts.update_latest_data import update_all_klines

    logger.info("=" * 60)
    logger.info("开始更新币池数据（后200名低流动性币种）")
    logger.info("=" * 60)

    try:
        daily_dir = ROOT / "data" / "daily_klines"
        hourly_dir = ROOT / "data" / "hourly_klines"

        update_all_klines(daily_dir, hourly_dir)
        logger.info("✓ 币池数据更新完成")
        return True

    except Exception as e:
        logger.error(f"✗ 币池数据更新失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_paper_trading():
    """运行Paper Trading"""
    from scripts.paper_trader import PaperTrader

    logger.info("=" * 60)
    logger.info("开始运行Paper Trading")
    logger.info("=" * 60)

    try:
        trader = PaperTrader(
            daily_dir=ROOT / "data" / "daily_klines",
            hourly_dir=ROOT / "data" / "hourly_klines",
            model_dir=ROOT / "models",
            output_dir=ROOT / "data" / "paper_trading"
        )

        # 使用当前日期
        current_date = datetime.now().strftime("%Y-%m-%d")
        trader.run(date=current_date, max_positions=20)

        logger.info("✓ Paper Trading 完成")
        return True

    except Exception as e:
        logger.error(f"✗ Paper Trading 失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def collect_signal_statistics():
    """收集信号统计数据"""
    logger.info("=" * 60)
    logger.info("收集信号统计数据")
    logger.info("=" * 60)

    try:
        signals_history_file = ROOT / "data" / "paper_trading" / "signals_history.csv"

        if not signals_history_file.exists():
            logger.warning("信号历史文件不存在，跳过统计")
            return None

        # 读取所有历史信号
        signals = pd.read_csv(signals_history_file)

        if len(signals) == 0:
            logger.warning("没有历史信号数据")
            return None

        # 解析时间
        signals['signal_time'] = pd.to_datetime(signals['signal_time'])
        signals['signal_date'] = signals['signal_time'].dt.date

        # 统计信息
        total_signals = len(signals)
        unique_symbols = signals['symbol'].nunique()
        date_range = (signals['signal_date'].min(), signals['signal_date'].max())

        # 按日期统计
        daily_counts = signals.groupby('signal_date').size()

        # 按模型分类统计（如果有）
        if 'model_class' in signals.columns:
            class_counts = signals['model_class'].value_counts().to_dict()
        else:
            class_counts = {}

        # 最近7天的信号
        last_7_days = datetime.now().date() - timedelta(days=7)
        recent_signals = signals[signals['signal_date'] >= last_7_days]

        stats = {
            'total_signals': total_signals,
            'unique_symbols': unique_symbols,
            'date_range': date_range,
            'daily_avg': daily_counts.mean(),
            'recent_7d_count': len(recent_signals),
            'class_distribution': class_counts,
            'signals_df': signals
        }

        logger.info(f"✓ 统计完成:")
        logger.info(f"  总信号数: {total_signals}")
        logger.info(f"  独立标的: {unique_symbols}")
        logger.info(f"  日期范围: {date_range[0]} ~ {date_range[1]}")
        logger.info(f"  日均信号: {daily_counts.mean():.1f}")
        logger.info(f"  最近7天: {len(recent_signals)} 个信号")

        if class_counts:
            logger.info(f"  模型分类: {class_counts}")

        return stats

    except Exception as e:
        logger.error(f"✗ 统计收集失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def check_and_collect_training_data(stats):
    """检查并收集训练数据（追踪信号后续表现）"""
    logger.info("=" * 60)
    logger.info("收集训练数据（追踪信号表现）")
    logger.info("=" * 60)

    try:
        if stats is None:
            logger.warning("无统计数据，跳过")
            return False

        signals = stats['signals_df']

        # 只处理5天前的信号（有足够的后续数据）
        cutoff_date = datetime.now().date() - timedelta(days=5)
        old_signals = signals[signals['signal_date'] <= cutoff_date].copy()

        if len(old_signals) == 0:
            logger.info("没有足够老的信号需要追踪")
            return False

        logger.info(f"找到 {len(old_signals)} 个可追踪的信号（>5天）")

        # 准备回测数据文件
        backtest_file = ROOT / "data" / "backtest_trades.csv"
        training_data_dir = ROOT / "data" / "training_samples"
        training_data_dir.mkdir(exist_ok=True)

        # 读取现有回测数据
        if backtest_file.exists():
            existing_trades = pd.read_csv(backtest_file)
            logger.info(f"现有回测交易: {len(existing_trades)} 条")
        else:
            existing_trades = pd.DataFrame()
            logger.info("创建新的回测数据文件")

        # 为每个信号计算实际表现（简化版）
        new_trades = []

        for _, signal in old_signals.iterrows():
            symbol = signal['symbol']
            signal_date = signal['signal_date']
            entry_price = signal['close']

            # 模拟计算5天后收益（这里需要实际K线数据）
            # 简化：使用历史数据计算
            try:
                daily_file = ROOT / "data" / "daily_klines" / f"{symbol}.csv"
                if daily_file.exists():
                    df = pd.read_csv(daily_file, parse_dates=['timestamp'])
                    df['date'] = df['timestamp'].dt.date

                    # 找到信号日期
                    signal_idx = df[df['date'] == signal_date].index
                    if len(signal_idx) == 0:
                        continue

                    signal_idx = signal_idx[0]

                    # 获取5天后的价格
                    if signal_idx + 5 < len(df):
                        exit_price = df.iloc[signal_idx + 5]['close']

                        # 做空收益
                        pnl_pct = (entry_price - exit_price) / entry_price

                        trade = {
                            'symbol': symbol,
                            'entry_date': signal_date,
                            'entry_price': entry_price,
                            'exit_date': df.iloc[signal_idx + 5]['date'],
                            'exit_price': exit_price,
                            'pnl_pct': pnl_pct,
                            'signal_type': 'SHORT',
                            'model_class': signal.get('model_class', None),
                            'model_score': signal.get('model_score', None)
                        }

                        new_trades.append(trade)

            except Exception as e:
                continue

        if len(new_trades) > 0:
            new_trades_df = pd.DataFrame(new_trades)

            # 合并到现有数据
            if len(existing_trades) > 0:
                all_trades = pd.concat([existing_trades, new_trades_df], ignore_index=True)
                # 去重
                all_trades = all_trades.drop_duplicates(
                    subset=['symbol', 'entry_date'],
                    keep='last'
                )
            else:
                all_trades = new_trades_df

            # 保存
            all_trades.to_csv(backtest_file, index=False)

            logger.info(f"✓ 新增 {len(new_trades)} 条交易记录")
            logger.info(f"✓ 总交易记录: {len(all_trades)} 条")
            logger.info(f"✓ 平均收益: {new_trades_df['pnl_pct'].mean():.2%}")

            return True
        else:
            logger.info("没有新的交易记录生成")
            return False

    except Exception as e:
        logger.error(f"✗ 训练数据收集失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def should_retrain_model(stats):
    """判断是否需要重新训练模型"""
    logger.info("=" * 60)
    logger.info("检查是否需要重新训练模型")
    logger.info("=" * 60)

    try:
        # 读取回测数据
        backtest_file = ROOT / "data" / "backtest_trades.csv"

        if not backtest_file.exists():
            logger.info("回测数据文件不存在，暂不训练")
            return False

        trades = pd.read_csv(backtest_file)

        if len(trades) < 100:
            logger.info(f"交易样本不足（{len(trades)}/100），暂不训练")
            return False

        # 检查上次训练时间
        model_file = ROOT / "models" / "rank_model.pt"
        model_meta_file = ROOT / "models" / "rank_model_meta.json"

        if not model_file.exists():
            logger.info("模型不存在，需要训练")
            return True

        # 检查模型文件修改时间
        model_mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
        hours_since_training = (datetime.now() - model_mtime).total_seconds() / 3600

        logger.info(f"模型训练时间: {model_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"距上次训练: {hours_since_training:.1f} 小时")

        # 判断条件：
        # 1. 超过24小时未训练
        # 2. 或者有足够多的新样本（比上次多50个）

        import json
        if model_meta_file.exists():
            with open(model_meta_file, 'r') as f:
                meta = json.load(f)
                last_train_samples = meta.get('train_samples', 0)
        else:
            last_train_samples = 0

        new_samples = len(trades) - last_train_samples

        if hours_since_training >= 24:
            logger.info(f"✓ 超过24小时未训练，需要重训")
            return True
        elif new_samples >= 50:
            logger.info(f"✓ 新增样本充足（{new_samples}个），需要重训")
            return True
        else:
            logger.info(f"暂不需要重训（新样本: {new_samples}, 距上次: {hours_since_training:.1f}h）")
            return False

    except Exception as e:
        logger.error(f"✗ 重训检查失败: {e}")
        return False


def retrain_model():
    """重新训练模型"""
    logger.info("=" * 60)
    logger.info("开始重新训练模型")
    logger.info("=" * 60)

    try:
        import subprocess

        # 准备训练参数
        cmd = [
            sys.executable,
            "-m", "scripts.modeling.train_ranker",
            "--candidates-dir", str(ROOT / "data" / "daily_scans"),
            "--backtest-csv", str(ROOT / "data" / "backtest_trades.csv"),
            "--daily-dir", str(ROOT / "data" / "daily_klines"),
            "--hourly-dir", str(ROOT / "data" / "hourly_klines"),
            "--output-dir", str(ROOT / "models"),
            "--epochs", "30",
            "--batch-size", "64",
            "--device", "cpu"
        ]

        logger.info(f"训练命令: {' '.join(cmd)}")

        # 运行训练（设置超时30分钟）
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=str(ROOT)
        )

        if result.returncode == 0:
            logger.info("✓ 模型训练成功")
            logger.info(f"训练输出:\n{result.stdout[-1000:]}")  # 最后1000字符
            return True
        else:
            logger.error(f"✗ 模型训练失败")
            logger.error(f"错误输出:\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("✗ 模型训练超时（30分钟）")
        return False
    except Exception as e:
        logger.error(f"✗ 模型训练异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def save_hourly_stats(stats):
    """保存每小时统计到文件"""
    try:
        stats_dir = ROOT / "data" / "statistics"
        stats_dir.mkdir(exist_ok=True)

        stats_file = stats_dir / f"stats_{datetime.now().strftime('%Y%m')}.csv"

        # 准备统计数据
        record = {
            'timestamp': datetime.now().isoformat(),
            'total_signals': stats['total_signals'] if stats else 0,
            'unique_symbols': stats['unique_symbols'] if stats else 0,
            'recent_7d_count': stats['recent_7d_count'] if stats else 0,
            'daily_avg': stats['daily_avg'] if stats else 0,
        }

        # 追加到CSV
        df = pd.DataFrame([record])

        if stats_file.exists():
            existing = pd.read_csv(stats_file)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_csv(stats_file, index=False)
        logger.info(f"✓ 统计数据已保存: {stats_file}")

    except Exception as e:
        logger.error(f"✗ 统计保存失败: {e}")


def main():
    """主流程"""
    start_time = datetime.now()
    logger.info("\n" + "=" * 80)
    logger.info(f"每小时自动交易（增强版）- 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    current_hour = datetime.now().hour

    # 步骤1: 更新币池数据（每天早上8点）
    if current_hour == 8:
        logger.info("\n[步骤1/5] 更新币池数据")
        data_success = update_coin_pool()
    else:
        logger.info("\n[步骤1/5] 跳过数据更新（非8点）")
        data_success = True

    # 步骤2: 运行Paper Trading（每小时）
    logger.info("\n[步骤2/5] 运行Paper Trading")
    trading_success = run_paper_trading()

    # 步骤3: 收集信号统计
    logger.info("\n[步骤3/5] 收集信号统计")
    stats = collect_signal_statistics()
    save_hourly_stats(stats)

    # 步骤4: 收集训练数据（追踪信号表现）
    logger.info("\n[步骤4/5] 收集训练数据")
    training_data_collected = check_and_collect_training_data(stats)

    # 步骤5: 检查并重新训练模型（如果需要）
    logger.info("\n[步骤5/5] 检查模型训练")
    model_retrained = False

    if should_retrain_model(stats):
        model_retrained = retrain_model()

    # 汇总
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\n" + "=" * 80)
    logger.info("运行完成")
    logger.info("=" * 80)
    logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总耗时: {duration:.1f} 秒")
    logger.info(f"数据更新: {'✓ 成功' if data_success and current_hour == 8 else '⊘ 跳过'}")
    logger.info(f"Paper Trading: {'✓ 成功' if trading_success else '✗ 失败'}")
    logger.info(f"统计收集: {'✓ 成功' if stats else '✗ 失败'}")
    logger.info(f"训练数据: {'✓ 已更新' if training_data_collected else '⊘ 无更新'}")
    logger.info(f"模型重训: {'✓ 已完成' if model_retrained else '⊘ 未执行'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
