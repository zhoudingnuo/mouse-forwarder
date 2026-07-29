"""
neural_aim.py - 在线学习神经网络瞄准模块 (X轴单输出)

替代 PID 控制器, 实时学习游戏灵敏度、延迟等非线性特性。
只输出 X 轴 (左右), 适合跟踪左右平移的 bot。

架构:
  输入 (2): error_x, vel_x
  → 全连接4 → ReLU → 全连接4 → ReLU → 全连接1 → 输出 (dx)

参数量: 2×4 + 4 + 4×4 + 4 + 4×1 + 1 = 8+4+16+4+4+1 = 37
"""

import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


class TinyNN:
    """
    微型神经网络, 在线学习瞄准映射 (X轴单输出)

    37 个参数, 推理 < 0.01ms
    """

    def __init__(self, lr: float = 0.01, max_step_px: float = 10.0):
        self.lr = lr
        self.max_step_px = max_step_px

        # 网络: 2 → 4 → 4 → 1
        # P型初始化: dx ≈ 0.25 * error_x
        self.W1 = np.array([[1.0, 0.5, 0.0, 0.0],   # error_x → hidden 0,1
                            [0.0, 0.0, 0.5, 0.25]])  # vel_x   → hidden 2,3
        self.b1 = np.zeros(4)

        self.W2 = np.eye(4) * 0.5
        self.W2[0, 1] = 0.2
        self.b2 = np.zeros(4)

        self.W3 = np.array([0.5, 0.3, 0.0, 0.0])  # [4] → 标量输出
        self.b3 = np.array([0.0])

        # 训练状态
        self._train_count = 0
        self._last_status: str = ''
        self._gain = 0.1

        # 性能评估
        self._error_history: list = []
        self._best_avg_error: float = 1e9
        self._best_weights: dict = {}
        self._saved_path: str = ''

        # 缓存 (用于反向传播)
        self._cache = {}

    def forward(self, x: np.ndarray) -> float:
        """
        前向推理

        Args:
            x: [2] = [error_x, vel_x]

        Returns:
            dx (float)
        """
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0)
        z3 = float(a2 @ self.W3 + self.b3)

        # 输出增益
        z3 = z3 * self._gain

        # NaN 防护
        if not np.isfinite(z3):
            z3 = 0.0

        self._cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3}
        return z3

    def train_step(self, error_x: float):
        """
        在线训练一步

        用当前误差方向和大小调整权重。

        Args:
            error_x: 当前帧的水平误差
        """
        cache = self._cache
        if not cache:
            return

        x = cache['x']
        a1 = cache['a1']
        a2 = cache['a2']
        output = cache['z3']

        # 计算目标输出
        target = output
        if abs(error_x) > 2.0:
            sign_err = 1.0 if error_x > 0 else -1.0
            sign_out = 1.0 if output > 0 else -1.0 if output < 0 else 0.0

            if sign_out == 0:
                target = sign_err * min(self.max_step_px, max(1.0, abs(error_x) * 0.08))
            elif sign_out != sign_err:
                target = sign_err * abs(output) * 0.8
            else:
                target = output * min(1.003, 1.0 + abs(error_x) / 2000.0)

        # 限幅
        delta = target - output
        delta = max(-3.0, min(3.0, delta))
        target = output + delta

        # 反向传播
        dL_dz3 = output - target  # 标量
        dL_dW3 = a2 * dL_dz3  # [4]
        dL_db3 = dL_dz3

        dL_da2 = self.W3 * dL_dz3  # [4]
        dL_dz2 = dL_da2 * (a2 > 0).astype(float)  # ReLU
        dL_dW2 = np.outer(a1, dL_dz2)  # [4,4]
        dL_db2 = dL_dz2  # [4]

        dL_da1 = dL_dz2 @ self.W2.T  # [4]
        dL_dz1 = dL_da1 * (a1 > 0).astype(float)
        dL_dW1 = np.outer(x, dL_dz1)  # [2,4]
        dL_db1 = dL_dz1  # [4]

        # SGD 更新 + 梯度裁剪
        for param, grad in [(self.W1, dL_dW1), (self.b1, dL_db1),
                            (self.W2, dL_dW2), (self.b2, dL_db2),
                            (self.W3, dL_dW3), (self.b3, dL_db3)]:
            param -= self.lr * np.clip(grad, -0.5, 0.5)

        # 权重裁剪
        for param in [self.W1, self.W2, self.W3]:
            np.clip(param, -5.0, 5.0, out=param)

        self._train_count += 1
        self._gain = min(1.0, 0.1 + 0.9 * self._train_count / 100000)

        # 性能评估
        self._error_history.append(abs(error_x))
        if len(self._error_history) > 60:
            self._error_history.pop(0)
        if self._train_count >= 60:
            avg_err = sum(self._error_history) / len(self._error_history)
            if avg_err < self._best_avg_error:
                self._best_avg_error = avg_err
                self._best_weights = {
                    'W1': self.W1.copy(), 'b1': self.b1.copy(),
                    'W2': self.W2.copy(), 'b2': self.b2.copy(),
                    'W3': self.W3.copy(), 'b3': self.b3.copy(),
                }
                logger.info(f'NN best avg_err={avg_err:.1f}px step={self._train_count}')
                if avg_err < 8.0 and self._gain > 0.5 and not self._saved_path:
                    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'nn_models')
                    os.makedirs(save_dir, exist_ok=True)
                    path = os.path.join(save_dir, f'nn_best_{self._train_count}s_{avg_err:.0f}px.npz')
                    self.save(path)
                    self._saved_path = path
                    logger.info(f'✅ NN saved {path}')

        # 每 30 帧打印状态
        if self._train_count % 30 == 0:
            w1_mean = np.abs(self.W1).mean()
            w3_mean = np.abs(self.W3).mean()
            avg_e = sum(self._error_history) / max(1, len(self._error_history))
            self._last_status = (f'🧠 train={self._train_count} gain={self._gain:.2f} '
                                 f'avg_err={avg_e:.1f}px best={self._best_avg_error:.1f}px '
                                 f'|W1|={w1_mean:.2f} |W3|={w3_mean:.2f}')

    def get_stats(self) -> dict:
        avg_err = sum(self._error_history) / max(1, len(self._error_history)) if self._error_history else 0
        return {
            'train_steps': self._train_count,
            'gain': round(self._gain, 3),
            'avg_error_60': round(avg_err, 1),
            'best_avg_error': round(self._best_avg_error, 1) if self._best_avg_error < 1e9 else 0,
            'saved_path': self._saved_path,
            'last_status': self._last_status,
        }

    def save(self, path: str):
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)

    def load(self, path: str):
        data = np.load(path)
        self.W1 = data['W1']; self.b1 = data['b1']
        self.W2 = data['W2']; self.b2 = data['b2']
        self.W3 = data['W3']; self.b3 = data['b3']
