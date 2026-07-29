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
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class TinyNN:
    """
    微型神经网络, 在线学习瞄准映射

    权重存在 numpy 数组中, 手动实现前向和反向传播。
    """

    def __init__(self, lr: float = 0.001, y_scale: float = 0.3):
        # 网络结构: 4 → 8 → 8 → 2
        # 输入: [error_x, error_y, vel_x, vel_y]
        # 输出: [dx, dy]
        self.lr = lr
        self.y_scale = y_scale

        # P型初始化: 初始行为 ≈ PID, 在线学习逐步优化
        # W1: error_x 走前4个神经元, error_y 走后4个
        self.W1 = np.zeros((4, 8))
        self.W1[0, 0:4] = 1.0  # error_x → hidden 0-3
        self.W1[1, 4:8] = 1.0  # error_y → hidden 4-7
        self.W1[2, 2] = 0.5  # vel_x 少量输入
        self.W1[3, 6] = 0.5  # vel_y 少量输入
        self.b1 = np.ones(8) * 0.1  # 小偏置保持 ReLU 激活

        # W2: 部分交叉混合
        self.W2 = np.eye(8) * 0.5
        self.W2[0, 4] = 0.2  # 少量 x→y 交叉
        self.W2[4, 0] = 0.2  # 少量 y→x 交叉
        self.b2 = np.zeros(8)

        # W3: 映射到输出, 初始 dx ≈ 0.3*error_x, dy ≈ 0.3*y_scale*error_y
        self.W3 = np.zeros((8, 2))
        self.W3[0, 0] = 0.3   # dx
        self.W3[1, 0] = 0.2
        self.W3[4, 1] = 0.3 * y_scale  # dy (带 y_scale 限制)
        self.W3[5, 1] = 0.2 * y_scale
        self.b3 = np.zeros(2)

        # 训练缓存
        self._buffer: deque = deque(maxlen=64)
        self._train_count = 0

        # 从上一次推理保存的中间值 (用于反向传播)
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向推理

        Args:
            x: [4] = [error_x, error_y, vel_x, vel_y]

        Returns:
            [2] = [dx, dy]  浮点数, 直接用作鼠标位移
        """
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0)
        z3 = a2 @ self.W3 + self.b3

        self._cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3}
        return z3  # 输出 (dx, dy)

    def train_step(self, error_now: float, error_before: float):
        """
        在线训练一步

        用当前的误差变化来估计"好"的输出方向。
        如果误差减小 → 强化当前输出方向
        如果误差增大 → 反向修正

        Args:
            error_now: 当前帧的误差距离
            error_before: 上一帧的误差距离
        """
        cache = self._cache
        if not cache:
            return

        x = cache['x']
        a2 = cache['a2']
        output = cache['z3']

        # 计算"伪标签": 误差减小时加强当前输出, 增大时反转
        error_delta = error_now - error_before
        # 强度: 误差变化越大, 学习步长越大
        intensity = min(1.0, abs(error_delta) / 20.0)

        if error_delta < 0:
            # 误差减小 → 当前方向是对的, 加强
            target = output * (1.0 + intensity * 0.1)
        elif error_delta > 0:
            # 误差增大 → 方向可能不对, 减弱或反转
            if abs(output[0]) > 0.1:
                target = output * (1.0 - intensity * 0.2)
            else:
                target = output
        else:
            target = output

        # 限制目标变化幅度
        delta = target - output
        max_delta = 2.0
        delta = np.clip(delta, -max_delta, max_delta)
        target = output + delta

        # 手动反向传播
        batch_size = 1

        # 输出层梯度
        dL_dz3 = (output - target) / batch_size  # [2]
        dL_dW3 = np.outer(a2, dL_dz3)  # [8,2]
        dL_db3 = dL_dz3  # [2]

        # 隐藏层2
        dL_da2 = dL_dz3 @ self.W3.T  # [8]
        dL_dz2 = dL_da2 * (a2 > 0).astype(float)  # ReLU 导数
        dL_dW2 = np.outer(cache['a1'], dL_dz2)  # [8,8]
        dL_db2 = dL_dz2  # [8]

        # 隐藏层1
        dL_da1 = dL_dz2 @ self.W2.T  # [8]
        dL_dz1 = dL_da1 * (cache['a1'] > 0).astype(float)
        dL_dW1 = np.outer(x, dL_dz1)  # [4,8]
        dL_db1 = dL_dz1  # [8]

        # SGD 更新
        self.W3 -= self.lr * dL_dW3
        self.b3 -= self.lr * dL_db3
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1

        self._train_count += 1
        self._buffer.append((abs(error_delta), intensity))

    def get_stats(self) -> dict:
        """获取训练统计"""
        avg_delta = np.mean([d for d, _ in self._buffer]) if self._buffer else 0
        return {
            'train_steps': self._train_count,
            'buffer_size': len(self._buffer),
            'avg_error_delta': round(avg_delta, 3),
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
