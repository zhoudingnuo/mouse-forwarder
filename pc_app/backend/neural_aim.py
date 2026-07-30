"""
neural_aim.py - 在线学习神经网络瞄准模块 (X轴单输出 + 历史步数)

架构:
  输入 (3): error_x, vel_x, prev_dx (上一步输出)
  → 全连接6 → 全连接6 → 全连接1 → 输出 dx

参数: 3×6+6 + 6×6+6 + 6×1+1 = 24+42+7 = 73
"""

import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


class TinyNN:
    def __init__(self, lr: float = 0.01, max_step_px: float = 10.0):
        self.lr = lr
        self.max_step_px = max_step_px
        self._prev_dx = 0.0
        self._prev_dx_clamped = 0.0  # 限幅后的上一步输出

        # 网络: 3 → 6 → 6 → 1 (全线性)
        # P型初始化: dx ≈ 0.25*error_x + 0.1*vel_x + 0.1*prev_dx
        self.W1 = np.zeros((3, 6))
        self.W1[0, :3] = [1.0, 0.5, 0.3]   # error_x
        self.W1[1, 3:5] = [0.5, 0.3]       # vel_x
        self.W1[2, 5] = 0.3                 # prev_dx
        self.b1 = np.zeros(6)

        self.W2 = np.eye(6) * 0.5
        self.W2[0, 3] = 0.2
        self.W2[3, 0] = 0.2
        self.W2[1, 5] = 0.1
        self.b2 = np.zeros(6)

        self.W3 = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0])  # [6]
        self.b3 = np.array([0.0])

        self._train_count = 0
        self._last_status = ''
        self._gain = 0.1
        self._best_score = 0.0       # 最佳综合得分 (100=完美)
        self._best_weights = {}
        self._saved_path = ''
        self._cache = {}

    def forward(self, x: np.ndarray) -> float:
        """前向: [error_x, vel_x] → dx (自动带上 prev_dx)"""
        x3 = np.array([x[0], x[1], self._prev_dx_clamped], dtype=np.float32)
        z1 = x3 @ self.W1 + self.b1
        z2 = z1 @ self.W2 + self.b2
        z3 = float(z2 @ self.W3 + self.b3) * self._gain

        if not np.isfinite(z3):
            z3 = 0.0

        self._cache = {'x3': x3, 'z1': z1, 'z2': z2, 'z3': z3}
        return z3

    def train_step(self, error_x: float):
        cache = self._cache
        if not cache:
            return

        x3 = cache['x3']
        z1 = cache['z1']
        z2 = cache['z2']
        output = cache['z3']

        # 保存上一步输出 (限幅, 防止爆炸)
        self._prev_dx = output
        self._prev_dx_clamped = max(-30.0, min(30.0, output))  # 不超过 ±30px

        # 训练目标
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

        delta = target - output
        delta = max(-3.0, min(3.0, delta))
        target = output + delta

        # 反向传播 (全线性)
        dL_dz3 = output - target
        dL_dW3 = z2 * dL_dz3
        dL_db3 = dL_dz3

        dL_dz2 = dL_dz3 * self.W3
        dL_dW2 = np.outer(z1, dL_dz2)
        dL_db2 = dL_dz2

        dL_dz1 = dL_dz2 @ self.W2.T
        dL_dW1 = np.outer(x3, dL_dz1)
        dL_db1 = dL_dz1

        for param, grad in [(self.W1, dL_dW1), (self.b1, dL_db1),
                            (self.W2, dL_dW2), (self.b2, dL_db2),
                            (self.W3, dL_dW3), (self.b3, dL_db3)]:
            param -= self.lr * np.clip(grad, -0.5, 0.5)

        for param in [self.W1, self.W2, self.W3]:
            np.clip(param, -5.0, 5.0, out=param)

        self._train_count += 1
        self._gain = min(1.0, 0.1 + 0.9 * self._train_count / 100000)

        # 评分: 误差得分 + 稳定性惩罚
        abs_err = abs(error_x)
        if abs_err <= 5:
            error_score = 100.0
        elif abs_err <= 10:
            error_score = 100.0 - 40.0 * (abs_err - 5) / 5.0  # 100→60
        elif abs_err <= 30:
            error_score = 60.0 - 60.0 * (abs_err - 10) / 20.0  # 60→0
        else:
            error_score = 0.0

        # 震荡检测: 输出方向频繁切换扣分
        if hasattr(self, '_prev_sign') and self._prev_sign != 0:
            curr_sign = 1 if output > 0 else -1 if output < 0 else 0
            if curr_sign != 0 and curr_sign != self._prev_sign:
                self._osc_count = getattr(self, '_osc_count', 0) + 1
            else:
                self._osc_count = max(0, getattr(self, '_osc_count', 0) - 2)
        self._prev_sign = 1 if output > 0 else -1 if output < 0 else 0

        osc_penalty = min(30, getattr(self, '_osc_count', 0) * 5)  # 每次震荡扣5分, 最多扣30
        total_score = max(0, error_score - osc_penalty)

        # 最佳得分 → 备份权重 + 自动保存 (2000步后开始)
        if self._train_count >= 2000 and total_score > self._best_score:
            self._best_score = total_score
            self._best_weights = {
                'W1': self.W1.copy(), 'b1': self.b1.copy(),
                'W2': self.W2.copy(), 'b2': self.b2.copy(),
                'W3': self.W3.copy(), 'b3': self.b3.copy(),
            }
            logger.info(f'NN new best score={total_score:.0f} step={self._train_count}')
            if total_score >= 80 and self._gain > 0.5 and not self._saved_path:
                save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'nn_models')
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f'nn_best_{self._train_count}s_{total_score:.0f}score.npz')
                self.save(path)
                self._saved_path = path
                logger.info(f'✅ NN saved {path}')

        if self._train_count % 30 == 0:
            w1_mean = np.abs(self.W1).mean()
            w3_mean = np.abs(self.W3).mean()
            self._last_status = (f'🧠 train={self._train_count} gain={self._gain:.2f} '
                                 f'score={total_score:.0f} best={self._best_score:.0f} '
                                 f'osc={getattr(self, "_osc_count", 0)} '
                                 f'|W1|={w1_mean:.2f} |W3|={w3_mean:.2f} '
                                 f'prev={self._prev_dx_clamped:.0f}')

    def get_stats(self) -> dict:
        return {
            'train_steps': self._train_count,
            'gain': round(self._gain, 3),
            'score': round(self._best_score, 0),
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
