@echo off
REM Windows 定时任务脚本 - 每小时运行一次Paper Trading（增强版）
REM 使用 Windows 任务计划程序调用此脚本

cd /d F:\2025\Quant4Little

echo ========================================
echo 每小时自动交易（增强版 - 含模型自动更新）
echo 时间: %date% %time%
echo ========================================

REM 激活虚拟环境（如果有）
REM call venv\Scripts\activate.bat

REM 运行Python脚本（增强版）
python scripts\hourly_trading_enhanced.py

echo.
echo 运行完成
echo ========================================
