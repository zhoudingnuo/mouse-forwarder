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

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryConfig:
    """轨迹计算配置"""
    enabled: bool = False               # 总开关
    smooth_factor: float = 0.40         # 平滑因子 (0-1), 越大越跟手
    max_step_px: int = 8                # 单步最大像素位移 (越小越稳)
    prediction_ticks: int = 5           # 目标预测帧数 (补偿推理延迟)
    target_offset_x: int = 0            # 瞄准偏移 X (目标框宽度的百分比, 如 5=5%)
    target_offset_y: int = 0            # 瞄准偏移 Y (目标框高度的百分比)
    min_confidence: float = 0.30        # 最低置信度阈值
    target_priority: int = 1           # 目标类别优先级 (1=head, -1=最近, 0=body, 2=teammate, 3=breakable, 4=dodge)
    jitter_amount: float = 0.10         # 随机抖动幅度 (像素)
    smoothing_samples: int = 8          # 指数移动平均的采样窗口
    max_trail_points: int = 200         # 轨迹点最大保留数
    fov_radius: int = 300               # 自瞄范围 (像素), 0=禁用范围限制
    target_scale_x: float = 1.0        # 缩放 (检测→目标, 默认 1:1)
    target_scale_y: float = 1.0        # 缩放
    invert_ai_x: bool = False          # 反转 AI 轨迹 X 轴
    invert_ai_y: bool = False          # 反转 AI 轨迹 Y 轴
    # PID 控制器参数
    kp: float = 0.35                    # 比例增益 (直接响应偏移)
    ki: float = 0.02                    # 积分增益 (消除稳态误差, 补偿灵敏度)
    kd: float = 0.10                    # 微分增益 (抑制过冲)
    velocity_ff: float = 0.0           # 速度前馈 (0=关闭)
    integral_limit: float = 100.0       # 积分限幅 (防积分饱和)
    nn_mode: bool = False              # True=神经网络控制, False=PID控制
    nn_lr: float = 0.001               # 神经网络学习率
    max_steps_per_frame: int = 30      # 每帧最多发送的步数
    settle_deadzone: float = 3.0        # 进入死区的阈值 (像素), 目标在此范围内停止移动 (最小1px)
    unsettle_hysteresis: float = 5.0   # 退出死区的滞回阈值 (像素), 防止抖动 (最小1px)
    y_scale: float = 0.60              # Y 轴幅度缩放 (0.0-1.0), 防抖用


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


@dataclass
class LogEntry:
    """单帧轨迹日志"""
    timestamp: float          # 系统时间
    target_cx: float          # 目标中心 X
    target_cy: float          # 目标中心 Y
    target_class: int         # 目标类别
    target_confidence: float  # 目标置信度
    error_x: float            # 距屏幕中心误差 X
    error_y: float            # 距屏幕中心误差 Y
    distance: float           # 误差距离
    output_x: float           # PID 输出 X
    output_y: float           # PID 输出 Y
    steps: str                # 发送的步序列 (json)
    settled: bool             # 是否已对准
    prediction_x: float       # 预测后的瞄准点 X
    prediction_y: float       # 预测后的瞄准点 Y
    dt: float                 # 距上一帧的时间


class TrajectoryCalculator:
    """
    轨迹计算器

    根据目标检测结果计算平滑瞄准轨迹,
    输出微小的 (dx, dy) 相对位移序列。
    """

    def __init__(self):
        self.config = TrajectoryConfig()

        # 神经网络控制器 (延迟初始化)
        self._nn = None

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

        # 锁定目标跟踪 (避免来回切换)
        self._locked_target_cx: Optional[float] = None
        self._locked_target_cy: Optional[float] = None
        self._locked_target_class: int = 0  # 锁定目标的 class_id
        self._locked_ref_cx: Optional[float] = None  # 锁定时的参考点 X (所有候选的中心)
        self._locked_ref_cy: Optional[float] = None  # 锁定时的参考点 Y
        self._lock_frames: int = 0

        # 到达目标后的冷却计数器 (防止抖动反复触发)
        self._settled_frames: int = 0
        # 目标丢失计数器 (模型偶尔漏检时保持锁定)
        self._missed_frames: int = 0

        # 当前选中的目标 (用于前端画框高亮)
        self.selected_target: Optional[Detection] = None
        self.aim_x: float = 0
        self.aim_y: float = 0
        self.is_settled: bool = False

        # PID 控制器状态
        self._integral_x: float = 0.0
        self._integral_y: float = 0.0
        self._prev_error_x: float = 0.0
        self._prev_error_y: float = 0.0
        self._prev_dist_for_nn: float = 0.0  # 上一帧误差, 用于 NN 在线训练

        # 统计
        self._trajectories_computed: int = 0
        self._targets_acquired: int = 0

        # 轨迹日志
        self._log_enabled: bool = False
        self._log_entries: List[LogEntry] = []
        self._last_log_time: float = 0.0

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
        计算瞄准轨迹 (PID 控制器 + 检测反馈)

        每帧直接从检测结果计算误差 (目标与屏幕中心的偏移),
        用 PID 公式计算输出位移:
          output = Kp * error + Ki * integral + Kd * derivative

        不追踪已发送位移, 依靠检测反馈自然收敛。
        积分项消除稳态误差 (补偿游戏鼠标灵敏度),
        微分项抑制过冲。

        Args:
            detections: 当前帧的检测结果列表

        Returns:
            [(dx, dy), ...] 微移动序列, 用于发送到板子
            空列表表示无目标或轨迹禁用
        """
        if not self.config.enabled or not detections:
            self._smooth_x = None
            self._smooth_y = None
            self.selected_target = None
            self.is_settled = False
            self._integral_x = 0.0
            self._integral_y = 0.0
            self._prev_error_x = 0.0
            self._prev_error_y = 0.0
            return []

        # 1. 选择目标
        target = self._select_target(detections)
        if target is None:
            self.selected_target = None
            self.is_settled = False
            self._integral_x = 0.0
            self._integral_y = 0.0
            self._prev_error_x = 0.0
            self._prev_error_y = 0.0
            return []

        self.selected_target = target
        self._targets_acquired += 1

        # 2. 目标位置预测
        predicted_x, predicted_y = self._predict_target(target.cx, target.cy)

        # 3. 应用瞄准偏移 (目标框尺寸的百分比)
        aim_x = predicted_x + target.w * self.config.target_offset_x / 100.0
        aim_y = predicted_y + target.h * self.config.target_offset_y / 100.0
        self.aim_x = aim_x
        self.aim_y = aim_y

        # 4. PID 控制器
        #    误差 = 目标 - 屏幕中心 (每帧从检测反馈重新获取)
        error_x = aim_x - self._screen_center_x
        error_y = aim_y - self._screen_center_y
        # Y轴误差按 y_scale 缩放后再计算距离, 使 Y 死区同步缩小
        scaled_ey = error_y * self.config.y_scale
        distance = math.sqrt(error_x ** 2 + scaled_ey ** 2)

        # ── 神经网络在线训练 (用上一帧误差和当前误差) ──
        if self.config.nn_mode and self._nn is not None and self._prev_dist_for_nn > 0:
            self._nn.train_step(distance, self._prev_dist_for_nn)
        self._prev_dist_for_nn = distance

# ── 滞回死区 (从配置读取) ──
        SETTLE_DEADZONE = self.config.settle_deadzone
        UNSETTLE_HYST = self.config.unsettle_hysteresis

        if self.is_settled:
            # 已对准状态: 用较大的滞回阈值
            if distance < UNSETTLE_HYST:
                # 保持对准, 不发步数
                return []
            else:
                # 超出滞回范围, 退出对准状态
                self.is_settled = False
                self._settled_frames = 0
        else:
            # 未对准: 检查是否进入死区
            if distance < SETTLE_DEADZONE:
                self._settled_frames += 1
                if self._settled_frames >= 3:
                    # 连续 3 帧在死区内 → 标记已对准
                    self.is_settled = True
                    self._integral_x = 0.0
                    self._integral_y = 0.0
                    self._prev_error_x = 0.0
                    self._prev_error_y = 0.0
                return []
            else:
                self._settled_frames = 0

        # ── 积分项: 带泄漏 + 近距清零 ──
        # 误差较小时清零积分, 防止积分饱和导致振荡
        if distance < 30.0:
            self._integral_x = 0.0
            self._integral_y = 0.0
        else:
            # 泄漏积分器: 每帧衰减 5%, 防止无限累积
            self._integral_x = self._integral_x * 0.95 + error_x
            self._integral_y = self._integral_y * 0.95 + error_y
            i_limit = self.config.integral_limit
            self._integral_x = max(-i_limit, min(i_limit, self._integral_x))
            self._integral_y = max(-i_limit, min(i_limit, self._integral_y))

        # ── 微分项: 误差变化率 ──
        deriv_x = error_x - self._prev_error_x
        deriv_y = error_y - self._prev_error_y
        self._prev_error_x = error_x
        self._prev_error_y = error_y

        # ── PID 输出 + 速度前馈 ──
        if self.config.nn_mode:
            # 神经网络模式
            if self._nn is None:
                from neural_aim import TinyNN
                self._nn = TinyNN(lr=self.config.nn_lr, y_scale=self.config.y_scale)
                logger.info("Neural network controller initialized")

            # 估计目标速度
            vx, vy = 0.0, 0.0
            if len(self._target_history) >= 3:
                recent = self._target_history[-3:]
                vx = (recent[-1][0] - recent[0][0]) / 2
                vy = (recent[-1][1] - recent[0][1]) / 2

            # NN 前向: [error_x, error_y, vel_x, vel_y] → [dx, dy]
            inp = np.array([error_x, error_y, vx, vy], dtype=np.float32)
            nn_out = self._nn.forward(inp)
            output_x, output_y = float(nn_out[0]), float(nn_out[1])
        else:
            # PID 模式
            kp = self.config.kp
            ki = self.config.ki
            kd = self.config.kd
            output_x = kp * error_x + ki * self._integral_x + kd * deriv_x
            output_y = (kp * error_y + ki * self._integral_y + kd * deriv_y) * self.config.y_scale

            # 速度前馈
            ff = self.config.velocity_ff
            if ff > 0 and len(self._target_history) >= 3:
                recent = self._target_history[-3:]
                vx = (recent[-1][0] - recent[0][0]) / 2
                vy = (recent[-1][1] - recent[0][1]) / 2
                output_x += ff * vx
                output_y += ff * vy

        # 应用符号反转
        if self.config.invert_ai_x:
            output_x = -output_x
        if self.config.invert_ai_y:
            output_y = -output_y

        # 5. 分解为微移动序列
        all_steps = self._decompose_movement(output_x, output_y)

        # 6. 限制每帧步数, 防止 USB 堆积
        max_steps = self.config.max_steps_per_frame
        if len(all_steps) > max_steps:
            all_steps = all_steps[:max_steps]

        # 7. 添加到轨迹历史
        self._add_trail_point(aim_x, aim_y)

        self._trajectories_computed += 1

        # 记录轨迹日志
        if self._log_enabled:
            now = time.time()
            dt = now - self._last_log_time if self._last_log_time > 0 else 0
            self._last_log_time = now
            import json
            self._log_entries.append(LogEntry(
                timestamp=now,
                target_cx=target.cx,
                target_cy=target.cy,
                target_class=target.class_id,
                target_confidence=target.confidence,
                error_x=error_x,
                error_y=error_y,
                distance=distance,
                output_x=output_x,
                output_y=output_y,
                steps=json.dumps(all_steps),
                settled=self.is_settled,
                prediction_x=aim_x,
                prediction_y=aim_y,
                dt=dt,
            ))

        return all_steps

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
        self._locked_target_cx = None
        self._locked_target_cy = None
        self._locked_ref_cx = None
        self._locked_ref_cy = None
        self._locked_target_class = 0
        self._lock_frames = 0
        self._settled_frames = 0
        self._missed_frames = 0
        self.selected_target = None
        self.is_settled = False
        self._integral_x = 0.0
        self._integral_y = 0.0
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        logger.info("Trail cleared")

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'trajectories_computed': self._trajectories_computed,
            'targets_acquired': self._targets_acquired,
            'trail_points': len(self._trail_points),
        }

    # ============ 轨迹日志 ============

    def start_logging(self):
        """开始记录轨迹日志"""
        self._log_enabled = True
        self._log_entries.clear()
        self._last_log_time = 0.0
        logger.info("Trajectory logging started")

    def stop_logging(self, save_path: str = '') -> str:
        """停止记录轨迹日志, 如有路径则自动保存"""
        self._log_enabled = False
        count = len(self._log_entries)
        saved = ''
        if save_path and count > 0:
            saved = self.save_log_csv(save_path)
        logger.info(f"Trajectory logging stopped ({count} entries, saved to {saved})")
        return saved

    def get_log_csv(self) -> str:
        """导出日志为 CSV"""
        import io
        buf = io.StringIO()
        buf.write("frame,timestamp,target_cx,target_cy,target_class,confidence,"
                  "error_x,error_y,distance,output_x,output_y,"
                  "aim_x,aim_y,steps,settled,dt\n")
        for i, e in enumerate(self._log_entries):
            buf.write(
                f"{i},{e.timestamp:.3f},{e.target_cx:.1f},{e.target_cy:.1f},"
                f"{e.target_class},{e.target_confidence:.3f},"
                f"{e.error_x:.1f},{e.error_y:.1f},{e.distance:.1f},"
                f"{e.output_x:.2f},{e.output_y:.2f},"
                f"{e.prediction_x:.1f},{e.prediction_y:.1f},"
                f"\"{e.steps}\",{int(e.settled)},{e.dt:.4f}\n"
            )
        return buf.getvalue()

    def save_log_csv(self, filepath: str) -> str:
        """保存日志 CSV 到文件, 返回实际路径"""
        csv = self.get_log_csv()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(csv)
        return filepath

    def clear_log(self):
        """清空日志"""
        self._log_entries.clear()
        self._last_log_time = 0.0

    # ============ 内部方法 ============

    def _select_target(self, detections: List[Detection]) -> Optional[Detection]:
        """
        选择瞄准目标

        策略:
        1. 过滤置信度、类别、FOV
        2. 锁定当前目标: 一旦锁定, 每帧找离锁定目标最近的候选 (而非离屏幕中心最近的)
        3. 解锁条件: 锁定目标连续 30 帧未出现, 或新目标比锁定目标近 50%+ 到中心
        """
        candidates = [
            d for d in detections
            if d.confidence >= self.config.min_confidence
        ]

        if not candidates:
            # 模型偶尔漏检: 目标可能还在, 保持锁定最多 15 帧
            if self._locked_target_cx is not None and self._lock_frames > 0:
                self._missed_frames += 1
                if self._missed_frames < 15:
                    return Detection(
                        x=self._locked_target_cx - 20,
                        y=self._locked_target_cy - 20,
                        w=40, h=40,
                        confidence=0.5,
                        class_id=self._locked_target_class,
                        cx=self._locked_target_cx,
                        cy=self._locked_target_cy,
                    )
            self._locked_target_cx = None
            self._locked_target_cy = None
            self._locked_ref_cx = None
            self._locked_ref_cy = None
            self._locked_target_class = 0
            self._lock_frames = 0
            self._missed_frames = 0
            return None

        # 按目标类别优先级过滤
        if self.config.target_priority >= 0:
            filtered = [d for d in candidates if d.class_id == self.config.target_priority]
            if filtered:
                candidates = filtered

        # 按 FOV 范围过滤
        if self.config.fov_radius > 0:
            fov_sq = self.config.fov_radius ** 2
            in_fov = [
                d for d in candidates
                if (d.cx - self._screen_center_x) ** 2 +
                   (d.cy - self._screen_center_y) ** 2 <= fov_sq
            ]
            if in_fov:
                candidates = in_fov
            else:
                self._locked_target_cx = None
                self._locked_target_cy = None
                self._locked_ref_cx = None
                self._locked_ref_cy = None
                self._locked_target_class = 0
                self._lock_frames = 0
                return None

        if not candidates:
            self._locked_target_cx = None
            self._locked_target_cy = None
            self._locked_ref_cx = None
            self._locked_ref_cy = None
            self._locked_target_class = 0
            self._lock_frames = 0
            return None

        # 目标锁定逻辑: 避免来回切换
        if self._locked_target_cx is not None and self._lock_frames > 0:
            # 计算当前帧所有候选的中心 (参考点), 用于补偿视角移动
            cur_ref_cx = sum(d.cx for d in candidates) / len(candidates)
            cur_ref_cy = sum(d.cy for d in candidates) / len(candidates)

            # 计算视角偏移量 = 当前参考点 - 锁定时的参考点
            view_dx = cur_ref_cx - self._locked_ref_cx
            view_dy = cur_ref_cy - self._locked_ref_cy

            # 预测锁定目标在当前帧的位置 = 锁定位置 + 视角偏移
            predict_cx = self._locked_target_cx + view_dx
            predict_cy = self._locked_target_cy + view_dy

            # 找候选列表中离预测位置最近的
            def dist_to_predicted(d):
                return (d.cx - predict_cx) ** 2 + (d.cy - predict_cy) ** 2
            nearest_to_predicted = min(candidates, key=dist_to_predicted)
            predicted_dist = (predict_cx - self._screen_center_x) ** 2 + \
                             (predict_cy - self._screen_center_y) ** 2

            # 如果最近目标(到中心)比锁定目标预测位置近很多(50%+)才切换
            closest = min(candidates, key=lambda d: (d.cx - self._screen_center_x) ** 2 + (d.cy - self._screen_center_y) ** 2)
            closest_dist = (closest.cx - self._screen_center_x) ** 2 + \
                           (closest.cy - self._screen_center_y) ** 2

            if closest_dist < predicted_dist * 0.5:
                # 切换到新目标, 更新参考点
                self._missed_frames = 0
                self._locked_target_cx = closest.cx
                self._locked_target_cy = closest.cy
                self._locked_target_class = closest.class_id
                self._locked_ref_cx = cur_ref_cx
                self._locked_ref_cy = cur_ref_cy
                self._lock_frames = 60
                self._smooth_x = None
                self._smooth_y = None
                return closest
            else:
                # 保持锁定, 更新锁定位置为最近的候选
                # 更新参考点 (跟踪视角变化)
                self._missed_frames = 0
                self._locked_target_cx = nearest_to_predicted.cx
                self._locked_target_cy = nearest_to_predicted.cy
                self._locked_target_class = nearest_to_predicted.class_id
                self._locked_ref_cx = cur_ref_cx
                self._locked_ref_cy = cur_ref_cy
                self._lock_frames = 60
                return nearest_to_predicted
        else:
            # 首次锁定: 选离屏幕中心最近的, 记录参考点
            self._missed_frames = 0
            closest = min(candidates, key=lambda d: (d.cx - self._screen_center_x) ** 2 + (d.cy - self._screen_center_y) ** 2)
            self._locked_target_cx = closest.cx
            self._locked_target_cy = closest.cy
            self._locked_target_class = closest.class_id
            self._locked_ref_cx = sum(d.cx for d in candidates) / len(candidates)
            self._locked_ref_cy = sum(d.cy for d in candidates) / len(candidates)
            self._lock_frames = 60
            return closest

    def _predict_target(self, cx: float, cy: float) -> Tuple[float, float]:
        """
        目标位置预测 (补偿推理延迟)

        基于历史位置做线性外推, 补偿检测延迟。
        使用最近 3 帧计算瞬时速度, 比长时间平均更灵敏。
        """
        self._target_history.append((cx, cy))
        if len(self._target_history) > 10:
            self._target_history.pop(0)

        if len(self._target_history) < 3:
            return cx, cy

        # 使用最近 3 帧计算瞬时速度 (比全历史平均更跟手)
        recent = self._target_history[-3:]
        dx_total = recent[-1][0] - recent[0][0]
        dy_total = recent[-1][1] - recent[0][1]

        # 每帧平均位移 (2 帧间隔)
        step_dx = dx_total / 2
        step_dy = dy_total / 2

        # 外推预测, 补偿推理延迟
        predict_x = cx + step_dx * self.config.prediction_ticks
        predict_y = cy + step_dy * self.config.prediction_ticks

        return predict_x, predict_y

    def _decompose_movement(self, dx: float, dy: float) -> List[Tuple[int, int]]:
        """
        动态步长方分解为多步

        步长随距离递减: 远时 3x base, 近时 0.5x base
        每步间隔 1ms, 在帧空闲时间 (~6ms) 内尽量多发。
        base = max_step_px (默认 10)
        """
        max_step = self.config.max_step_px
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance == 0:
            return []

        # 动态步长倍数
        FAR_DIST = 100.0
        NEAR_DIST = 5.0
        if distance >= FAR_DIST:
            multiplier = 3.0
        elif distance <= NEAR_DIST:
            multiplier = 0.5
        else:
            t = (distance - NEAR_DIST) / (FAR_DIST - NEAR_DIST)
            multiplier = 0.5 + t * 2.5

        step_size = max_step * multiplier

        # 归一化方向
        norm_x = dx / distance
        norm_y = dy / distance

        jitter = self.config.jitter_amount
        steps = []
        remaining = distance

        while remaining > 0:
            # 当前步长: 越靠近越小
            if remaining >= FAR_DIST:
                m = 3.0
            elif remaining <= NEAR_DIST:
                m = 0.5
            else:
                t = (remaining - NEAR_DIST) / (FAR_DIST - NEAR_DIST)
                m = 0.5 + t * 2.5
            cur_step = max_step * m

            # 最后一步不超过剩余距离
            if cur_step > remaining:
                cur_step = remaining

            # 方向分量
            sdx = norm_x * cur_step
            sdy = norm_y * cur_step

            # 抖动 (靠近目标时衰减)
            near_factor = min(1.0, remaining / 50.0)
            jx = random.uniform(-jitter, jitter) * near_factor
            jy = random.uniform(-jitter, jitter) * near_factor

            sdx_int = max(-127, min(127, int(round(sdx + jx))))
            sdy_int = max(-127, min(127, int(round(sdy + jy))))

            if sdx_int != 0 or sdy_int != 0:
                steps.append((sdx_int, sdy_int))

            remaining -= cur_step
            if len(steps) >= 20:  # 安全上限, 避免死循环
                break

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