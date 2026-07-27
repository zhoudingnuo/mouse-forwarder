"""
trajectory_calculator.py - 瞄准轨迹计算模块

根据目标检测结果, 计算平滑的瞄准轨迹,
输出为一系列微小的相对位移 (dx, dy), 用于:
1. 通过串口发送到 CH32V305 硬件
2. 在前端 Canvas 上叠加可视化轨迹

轨迹算法:
- 选择目标: 离屏幕中心最近的高置信度目标
- 目标预测: 基于历史位置做线性预测
- 路径平滑: 指数移动平均 + 贝塞尔插值
- 反检测: 随机微扰动 + 步长限制
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryConfig:
    """轨迹计算配置"""
    enabled: bool = False               # 总开关
    smooth_factor: float = 0.35         # 平滑因子 (0-1), 越大越跟手
    max_step_px: int = 10               # 单步最大像素位移
    prediction_ticks: int = 3           # 目标预测帧数
    target_offset_x: int = 0            # 瞄准偏移 X (像素)
    target_offset_y: int = 0            # 瞄准偏移 Y (像素)
    min_confidence: float = 0.45        # 最低置信度阈值
    target_priority: int = 0            # 目标类别优先级 (-1=最近, 0=所有)
    jitter_amount: float = 0.15         # 随机抖动幅度 (像素)
    smoothing_samples: int = 5          # 指数移动平均的采样窗口
    max_trail_points: int = 200         # 轨迹点最大保留数


@dataclass
class Detection:
    """目标检测结果"""
    x: float            # 边界框左上角 X (绝对坐标)
    y: float            # 边界框左上角 Y (绝对坐标)
    w: float            # 宽度
    h: float            # 高度
    confidence: float   # 置信度 (0-1)
    class_id: int       # 类别 ID
    cx: float           # 中心点 X
    cy: float           # 中心点 Y

    @classmethod
    def from_bbox(cls, x, y, w, h, confidence, class_id):
        """从边界框创建"""
        return cls(
            x=x, y=y, w=w, h=h,
            confidence=confidence,
            class_id=class_id,
            cx=x + w / 2,
            cy=y + h / 2,
        )


@dataclass
class TrajectoryPoint:
    """轨迹点"""
    x: float
    y: float
    timestamp: float
    is_ai: bool = True  # True=AI轨迹, False=真实鼠标轨迹


class TrajectoryCalculator:
    """
    轨迹计算器

    根据目标检测结果计算平滑瞄准轨迹,
    输出微小的 (dx, dy) 相对位移序列。
    """

    def __init__(self):
        self.config = TrajectoryConfig()

        # 轨迹点历史 (用于前端可视化)
        self._trail_points: List[TrajectoryPoint] = []

        # 目标位置历史 (用于预测)
        self._target_history: List[Tuple[float, float]] = []

        # 当前平滑位置 (指数移动平均)
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None

        # 屏幕中心 (由外部设置)
        self._screen_center_x: float = 960
        self._screen_center_y: float = 540

        # 当前鼠标位置 (由外部更新)
        self._mouse_x: float = 960
        self._mouse_y: float = 540

        # 统计
        self._trajectories_computed: int = 0
        self._targets_acquired: int = 0

    def set_config(self, config: TrajectoryConfig):
        """更新配置"""
        self.config = config
        logger.info(f"Trajectory config updated: enabled={config.enabled}, "
                     f"smooth={config.smooth_factor}, max_step={config.max_step_px}")

    def set_screen_center(self, cx: float, cy: float):
        """设置屏幕中心坐标"""
        self._screen_center_x = cx
        self._screen_center_y = cy

    def update_mouse_position(self, x: float, y: float):
        """更新当前鼠标位置 (由鼠标事件回调调用)"""
        self._mouse_x = x
        self._mouse_y = y

    def calculate(self, detections: List[Detection]) -> List[Tuple[int, int]]:
        """
        计算瞄准轨迹

        Args:
            detections: 当前帧的检测结果列表

        Returns:
            [(dx, dy), ...] 微移动序列, 用于发送到板子
            空列表表示无目标或轨迹禁用
        """
        if not self.config.enabled or not detections:
            self._smooth_x = None
            self._smooth_y = None
            return []

        # 1. 选择目标
        target = self._select_target(detections)
        if target is None:
            return []

        self._targets_acquired += 1

        # 2. 目标位置预测
        predicted_x, predicted_y = self._predict_target(target.cx, target.cy)

        # 3. 应用瞄准偏移
        aim_x = predicted_x + self.config.target_offset_x
        aim_y = predicted_y + self.config.target_offset_y

        # 4. 计算相对位移
        raw_dx = aim_x - self._mouse_x
        raw_dy = aim_y - self._mouse_y

        # 如果已经在目标附近, 不做移动
        distance = math.sqrt(raw_dx ** 2 + raw_dy ** 2)
        if distance < 2.0:
            return []

        # 5. 指数移动平均平滑
        if self._smooth_x is None:
            self._smooth_x = raw_dx
            self._smooth_y = raw_dy
        else:
            alpha = self.config.smooth_factor
            self._smooth_x = alpha * raw_dx + (1 - alpha) * self._smooth_x
            self._smooth_y = alpha * raw_dy + (1 - alpha) * self._smooth_y

        smooth_dx = self._smooth_x
        smooth_dy = self._smooth_y

        # 6. 分解为微移动序列
        steps = self._decompose_movement(smooth_dx, smooth_dy)

        # 7. 添加到轨迹历史 (用于前端可视化)
        self._add_trail_point(aim_x, aim_y)

        self._trajectories_computed += 1

        return steps

    def add_real_mouse_point(self, dx: int, dy: int):
        """添加真实鼠标轨迹点 (用于可视化对比)"""
        self._trail_points.append(TrajectoryPoint(
            x=self._mouse_x + dx,
            y=self._mouse_y + dy,
            timestamp=time.time(),
            is_ai=False,
        ))
        self._prune_trail()

    def get_trail_points(self) -> List[TrajectoryPoint]:
        """获取所有轨迹点 (用于前端可视化)"""
        return self._trail_points.copy()

    def clear_trail(self):
        """清空轨迹"""
        self._trail_points.clear()
        self._target_history.clear()
        self._smooth_x = None
        self._smooth_y = None
        logger.info("Trail cleared")

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'trajectories_computed': self._trajectories_computed,
            'targets_acquired': self._targets_acquired,
            'trail_points': len(self._trail_points),
        }

    # ============ 内部方法 ============

    def _select_target(self, detections: List[Detection]) -> Optional[Detection]:
        """
        选择瞄准目标

        策略: 选择离屏幕中心最近且置信度达标的检测目标
        """
        candidates = [
            d for d in detections
            if d.confidence >= self.config.min_confidence
        ]

        if not candidates:
            return None

        # 如果设定了目标类别优先级, 过滤出指定类别
        if self.config.target_priority >= 0:
            # 无过滤, 全部候选
            pass

        # 选择离屏幕中心最近的目标
        best = min(
            candidates,
            key=lambda d: (
                (d.cx - self._screen_center_x) ** 2 +
                (d.cy - self._screen_center_y) ** 2
            )
        )

        return best

    def _predict_target(self, cx: float, cy: float) -> Tuple[float, float]:
        """
        目标位置预测

        基于历史位置做线性外推, 补偿检测延迟
        """
        self._target_history.append((cx, cy))
        if len(self._target_history) > 10:
            self._target_history.pop(0)

        if len(self._target_history) < 3:
            return cx, cy

        # 计算运动矢量
        dx_total = self._target_history[-1][0] - self._target_history[0][0]
        dy_total = self._target_history[-1][1] - self._target_history[0][1]
        n = len(self._target_history) - 1

        if n == 0:
            return cx, cy

        # 每帧平均位移
        step_dx = dx_total / n
        step_dy = dy_total / n

        # 外推预测
        predict_x = cx + step_dx * self.config.prediction_ticks
        predict_y = cy + step_dy * self.config.prediction_ticks

        return predict_x, predict_y

    def _decompose_movement(self, dx: float, dy: float) -> List[Tuple[int, int]]:
        """
        将大幅度移动分解为多个微移动

        每步不超过 max_step_px, 并添加随机抖动
        """
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance == 0:
            return []

        max_step = self.config.max_step_px
        jitter = self.config.jitter_amount

        # 计算需要的步数
        num_steps = max(1, int(math.ceil(distance / max_step)))

        # 归一化方向向量
        step_dx = dx / num_steps
        step_dy = dy / num_steps

        steps = []
        for i in range(num_steps):
            # 是否为最后一步 (最后一步不添加抖动, 确保精确)
            is_last = (i == num_steps - 1)

            if is_last:
                sdx = step_dx
                sdy = step_dy
            else:
                # 添加随机抖动 (反检测)
                jx = random.uniform(-jitter, jitter)
                jy = random.uniform(-jitter, jitter)
                sdx = step_dx + jx
                sdy = step_dy + jy

            # 舍入为整数 (串口协议要求 int8)
            sdx_int = max(-127, min(127, int(round(sdx))))
            sdy_int = max(-127, min(127, int(round(sdy))))

            # 跳过零位移
            if sdx_int == 0 and sdy_int == 0:
                continue

            steps.append((sdx_int, sdy_int))

        return steps

    def _add_trail_point(self, x: float, y: float):
        """添加轨迹点并裁剪"""
        self._trail_points.append(TrajectoryPoint(
            x=x, y=y,
            timestamp=time.time(),
            is_ai=True,
        ))
        self._prune_trail()

    def _prune_trail(self):
        """裁剪轨迹点到最大长度"""
        max_points = self.config.max_trail_points
        if len(self._trail_points) > max_points:
            self._trail_points = self._trail_points[-max_points:]