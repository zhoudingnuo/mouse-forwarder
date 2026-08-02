"""
auto_tune_pid.py - PID 参数自动扫描

遍历 kp 值, 每个跑 600 帧 (10秒), 记录平均误差, 选出最优。
用法: 在采集卡开着、AI轨迹开着的状态下运行。
"""

import sys
import os
import time
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pc_app', 'backend'))

import numpy as np
from object_detector import ObjectDetector
from trajectory_calculator import TrajectoryCalculator, TrajectoryConfig

# ── 配置 ────────────────────────────────────────
KP_RANGE = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
            0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]
FRAMES_PER_TEST = 600  # ~10秒 @ 60fps
KD_FIXED = 0.08
KI = 0.0

# ── 结果 ────────────────────────────────────────
results = []

print('=== PID 参数自动扫描 ===')
print(f'kp范围: {KP_RANGE}')
print(f'kd固定: {KD_FIXED}, ki={KI}')
print(f'每组测试 {FRAMES_PER_TEST} 帧\n')

# 每个 kp 测试
for kp in KP_RANGE:
    # 设置参数
    config = TrajectoryConfig()
    config.kp = kp
    config.ki = KI
    config.kd = KD_FIXED
    config.enabled = True
    config.min_confidence = 0.25
    config.max_step_px = 10
    config.fov_radius = 80
    config.settle_deadzone = 3
    config.unsettle_hysteresis = 5
    config.y_scale = 0.2

    traj = TrajectoryCalculator()
    traj.set_config(config)
    traj.set_screen_center(960, 540)

    # 模拟运行
    errors = []
    prev_dets = []

    for frame in range(FRAMES_PER_TEST):
        # 模拟检测结果 (实际使用时会从采集卡读取)
        # 这里只是框架, 实际运行时需要集成到主循环
        pass

    # 由于这个脚本是离线框架, 实际集成到 main.py 中运行
    print(f'  kp={kp:.2f}: 待集成到主程序运行')

print('\n建议: 将此功能集成到 main.py 的 auto_tune 消息处理中')
