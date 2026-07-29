"""
neural_aim.py - 在线学习神经网络瞄准模块

替代 PID 控制器, 实时学习游戏灵敏度、延迟等非线性特性。

架构:
  输入 (4): error_x, error_y, vel_x, vel_y
  → 全连接8 → ReLU → 全连接8 → ReLU → 全连接2 → 输出 (dx, dy)

在线训练:
  每帧前向推理输出 (dx, dy), 检测误差变化后做一步 SGD 更新。
  训练目标: 输出与"理想方向"的余弦相似度 + 幅度匹配。
"""

import logging
import math
import random
import time
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TinyNN:
    """
    微型神经网络, 在线学习瞄准映射

    权重存在 numpy 数组中, 手动实现前向和反向传播。
    """

    def __init__(self, lr: float = 0.01, y_scale: float = 0.3, max_step_px: float = 10.0):
        # 网络结构: 4 → 8 → 8 → 2 (全线性, 无激活函数)
        # 输入: [error_x, error_y, vel_x, vel_y]
        # 输出: [dx, dy]
        self.lr = lr
        self.y_scale = y_scale
        self.max_step_px = max_step_px

        # P型初始化: dx ≈ 0.25*error_x, dy ≈ 0.25*y_scale*error_y
        # W1: error_x → hidden[0:4], error_y → hidden[4:8]
        self.W1 = np.zeros((4, 8))
        self.W1[0, 0:4] = 1.0   # error_x
        self.W1[1, 4:8] = 1.0   # error_y
        self.W1[2, 2] = 0.5     # vel_x
        self.W1[3, 6] = 0.5     # vel_y
        self.b1 = np.zeros(8)

        # W2: 对角占优
        self.W2 = np.eye(8) * 0.5
        self.W2[0, 4] = 0.2
        self.W2[4, 0] = 0.2
        self.b2 = np.zeros(8)

        # W3: dx←hidden[0], dy←hidden[4]
        self.W3 = np.zeros((8, 2))
        self.W3[0, 0] = 0.5           # dx
        self.W3[1, 0] = 0.3
        self.W3[4, 1] = 0.5 * y_scale  # dy
        self.W3[5, 1] = 0.3 * y_scale
        self.b3 = np.zeros(2)

        # 训练缓存
        self._train_count = 0
        self._last_status: str = ''

        # 从上一次推理保存的中间值 (用于反向传播)
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向推理 (全线性网络, 无 ReLU, 保留正负号)"""
        z1 = x @ self.W1 + self.b1
        z2 = z1 @ self.W2 + self.b2
        z3 = z2 @ self.W3 + self.b3

        # NaN/Inf 防护 (权重发散时输出零)
        if not np.all(np.isfinite(z3)):
            z3 = np.zeros_like(z3)

        self._cache = {'x': x, 'z1': z1, 'z2': z2, 'z3': z3}
        return z3  # 输出 (dx, dy)

    def train_step(self, error_now: float, error_before: float):
        """
        在线训练一步 (基于误差方向, 而非误差变化幅度)

        - 输出方向正确 (与误差同号) → 微幅增强
        - 输出方向错误 → 反号
        - 输出为零 → 补一个最小步长
        """
        cache = self._cache
        if not cache:
            return

        x = cache['x']
        z1 = cache['z1']
        z2 = cache['z2']
        output = cache['z3']

        error_x = x[0]  # 当前 error_x
        output_x = output[0]

        # 计算目标输出方向
        target = output.copy()

        # X 轴: 根据误差方向调整
        if abs(error_x) > 2.0:  # 只有误差足够大时才训练
            sign_error = 1.0 if error_x > 0 else -1.0
            sign_output = 1.0 if output_x > 0 else -1.0 if output_x < 0 else 0.0

            if sign_output == 0:
                # 输出为零, 但应该有输出 → 给一个基础步长 (保守)
                target[0] = sign_error * min(max_step_px, max(1.0, abs(error_x) * 0.08))
            elif sign_output != sign_error:
                # 方向反了 → 翻转到正确方向
                target[0] = sign_error * abs(output_x) * 0.8
            else:
                # 方向正确 → 轻微增强 (每帧最多 0.3%)
                target[0] = output_x * min(1.003, 1.0 + abs(error_x) / 2000.0)

        # Y 轴同理 (带 y_scale 限制)
        if abs(x[1]) > 2.0:
            sign_err_y = 1.0 if x[1] > 0 else -1.0
            sign_out_y = 1.0 if output[1] > 0 else -1.0 if output[1] < 0 else 0.0
            if sign_out_y == 0:
                target[1] = sign_err_y * min(self.max_step_px, max(1.0, abs(x[1]) * 0.08)) * self.y_scale
            elif sign_out_y != sign_err_y:
                target[1] = sign_err_y * abs(output[1]) * 0.8
            else:
                target[1] = output[1] * min(1.003, 1.0 + abs(x[1]) / 2000.0)

        # 限制目标变化幅度
        delta = target - output
        delta = np.clip(delta, -3.0, 3.0)
        target = output + delta

        # 手动反向传播 (全线性, 无激活函数)
        dL_dz3 = (output - target)  # [2]

        # 输出层
        dL_dW3 = np.outer(z2, dL_dz3)  # [8,2]
        dL_db3 = dL_dz3  # [2]

        # 隐藏层2 (线性)
        dL_dz2 = dL_dz3 @ self.W3.T  # [8]
        dL_dW2 = np.outer(z1, dL_dz2)  # [8,8]
        dL_db2 = dL_dz2  # [8]

        # 隐藏层1 (线性)
        dL_dz1 = dL_dz2 @ self.W2.T  # [8]
        dL_dW1 = np.outer(x, dL_dz1)  # [4,8]
        dL_db1 = dL_dz1  # [8]

        # SGD 更新 (带梯度裁剪, 防止发散)
        for param, grad in [(self.W1, dL_dW1), (self.b1, dL_db1),
                            (self.W2, dL_dW2), (self.b2, dL_db2),
                            (self.W3, dL_dW3), (self.b3, dL_db3)]:
            # 梯度裁剪: 单元素最大变化 0.5
            grad_clipped = np.clip(grad, -0.5, 0.5)
            param -= self.lr * grad_clipped

        # 权重裁剪: 防止极端值
        for param in [self.W1, self.W2, self.W3]:
            np.clip(param, -5.0, 5.0, out=param)

        self._train_count += 1

        # 每 30 帧记录训练统计 (由 get_stats 返回)
        if self._train_count % 30 == 0:
            w1_mean = np.abs(self.W1).mean()
            w3_mean = np.abs(self.W3).mean()
            self._last_status = (f'🧠 train={self._train_count} '
                                 f'|W1|={w1_mean:.2f} |W3|={w3_mean:.2f} '
                                 f'out=({output[0]:.1f},{output[1]:.1f})')

    def get_stats(self) -> dict:
        """获取训练统计"""
        return {
            'train_steps': self._train_count,
            'last_status': self._last_status,
        }

    def save(self, path: str):
        """保存权重"""
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)

    def load(self, path: str):
        """加载权重"""
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']
