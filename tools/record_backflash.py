"""
record_backflash.py - 录制手动背闪轨迹

用途: 手动执行背闪动作 (快速甩动视角 ~90° 再拉回), 录制鼠标移动轨迹,
     供主程序自动背闪回放使用。

本脚本会:
    1. 自动连接 CH32V305 串口 (找到则连接)
    2. 把本地鼠标移动/点击/滚轮原样转发到串口 (游戏里的操作正常生效)
    3. 实时识别背闪动作 (快速甩动 + 拉回), 边录边学

识别模型 (由真实数据校准, tools/backflash_raw.json):
    - 鼠标事件先按方向聚合成"方向段"; |dx|<15 视为噪声并入
    - 反向或活跃事件间隔 >500ms 时切段 (点击微动/慢漂移自然隔开)
    - 一次背闪 = 主摆段 + 回摆段配对:
        主摆: 单向累计 >=800, 峰值事件 >=100, 时长 <=500ms (快速大幅甩动)
        回摆: 反向, 累计 >= max(500, 主摆*0.45), 紧接主摆 (间隔<=300ms)
    - 配对成功即输出一段; 幅度不够的段 (点击/微动) 自动丢弃

用法:
    python tools/record_backflash.py            # 正常录制 (识别后输出学习轨迹)
    python tools/record_backflash.py --raw      # 仅录原始数据到 backflash_raw.json (供分析)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pc_app', 'backend'))

from raw_mouse import RawMouseMonitor
from serial_forwarder import SerialForwarder
from protocol import encode_packet

TARGET_SEGMENTS = 8       # 录制段数 (背闪次数), 达到后自动结束 (Ctrl+C 可提前)

# ---- 识别阈值 (真实数据校准: 动作总时长 250~370ms, 主摆累计 1800~3000) ----
NOISE = 15                # |dx| 低于此 = 噪声 (并入当前段, 不算活跃事件)
GAP_ACT = 500             # 活跃事件间隔超此值 -> 切段 (动作间停顿/慢漂移隔离)
MIN_MAIN = 800            # 主摆累计位移下限 (真转身 vs 点击微动: 点击累计 ~150-300)
MAIN_PEAK = 100           # 主摆内单事件峰值下限 (快速甩动单事件 150~300)
MAIN_DUR_MS = 500         # 主摆时长上限
MIN_BACK_RATIO = 0.45     # 回摆累计 >= 主摆 * 此比例 (确认"有回来")
MIN_BACK_MS = 500         # 回摆累计绝对下限
MAX_PAIR_GAP_MS = 300     # 回摆起点距主摆终点的最大间隔
MAX_TOTAL_MS = 1100       # 整个动作 (主摆起~回摆止) 时长上限

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'backflash_traj.json')
RAW_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'backflash_raw.json')

RESIDUAL_RATIO = 0.05   # 回正补偿后允许残留的视角差 (相对最远转角)


def _split_move(dx: int, dy: int, max_step: int = 120):
    """把位移拆分为 int8 范围内的多步 (保持方向)"""
    steps = []
    while dx != 0 or dy != 0:
        sx = max(-max_step, min(max_step, dx))
        sy = max(-max_step, min(max_step, dy))
        steps.append((sx, sy))
        dx -= sx
        dy -= sy
    return steps


class BackflashRecorder:
    def __init__(self):
        self._monitor = None
        self._serial = None
        self._segments = []       # 完成的背闪段: [(t_ms, dx, dy), ...]
        self._all_events = []     # 本次运行的原始事件流 (raw 模式用)
        self._t0 = None
        self._finished = False
        self._discard_count = 0
        self._forward = False    # 是否转发到串口 (游戏视角会动)
        self._btn_state = 0      # 自维护的按钮状态 (按下/抬起沿更新, 与主程序一致)

        # ---- 增量识别状态机 ----
        self._cur = []            # 正在聚合的方向段 (含噪声事件)
        self._cur_dir = 0         # 当前段方向 (+1/-1)
        self._cur_last_act = 0.0  # 当前段最后活跃事件时间
        self._pend = None         # 最近闭合、符合"主摆"条件的段 {evs, s, e, cum, peak, net}

    # ================================================================
    # 鼠标事件 (转发 + 喂给识别器)
    # ================================================================
    def _on_event(self, e: dict):
        dx = int(e.get('dx', 0) or 0)
        dy = int(e.get('dy', 0) or 0)
        wheel = int(e.get('wheel', 0) or 0)

        # 按钮沿来自 raw 的 usButtonFlags; 快速点击 down+up 可能合并进同一事件,
        # 合成状态检测不到变化, 必须每个沿立即转发 (按下包 -> 抬起包)
        btn_down = int(e.get('btn_down', 0) or 0)
        btn_up = int(e.get('btn_up', 0) or 0)

        if dx == 0 and dy == 0 and wheel == 0 and not (btn_down or btn_up):
            return
        now_ms = (time.time() - self._t0) * 1000

        # 移动事件: 记录 + 喂识别器 (按钮/滚轮只转发不记录)
        if dx != 0 or dy != 0:
            self._all_events.append((now_ms, dx, dy))
            self._feed(now_ms, dx, dy)

        # 转发到串口: 按钮沿/滚轮/移动全发, 让游戏内点击和视角都正常
        if self._forward and self._serial and self._serial.is_connected:
            try:
                # 1) 按钮沿: 更新状态后立即发出 (合并事件也按 按下->抬起 顺序)
                if btn_down:
                    self._btn_state |= btn_down
                    self._serial._serial.write(encode_packet(self._btn_state, 0, 0, 0))
                if btn_up:
                    self._btn_state &= ~btn_up
                    self._serial._serial.write(encode_packet(self._btn_state, 0, 0, 0))
                # 2) 滚轮
                if wheel:
                    self._serial._serial.write(encode_packet(self._btn_state, 0, 0, wheel))
                # 3) 移动 (拆步, 带当前按钮状态: 按住扫射时保持按下)
                for sx, sy in _split_move(dx, dy, 120):
                    self._serial._serial.write(encode_packet(self._btn_state, sx, sy, 0))
            except Exception:
                pass

    # ================================================================
    # 增量识别: 方向段聚合 + 主摆/回摆配对
    # ================================================================
    def _feed(self, t: float, dx: int, dy: int):
        """喂入一个移动事件 (实时逐事件调用)"""
        if abs(dx) < NOISE:
            # 噪声并入当前段 (不延长活跃时间), 无段时忽略
            if self._cur:
                self._cur.append((t, dx, dy))
            return
        sdir = 1 if dx > 0 else -1
        if not self._cur:
            self._cur = [(t, dx, dy)]
            self._cur_dir = sdir
            self._cur_last_act = t
        elif sdir == self._cur_dir and t - self._cur_last_act <= GAP_ACT:
            self._cur.append((t, dx, dy))
            self._cur_last_act = t
        else:
            # 方向反转或停顿超时 -> 闭合当前段, 评估配对
            self._close_seg(self._cur)
            self._cur = [(t, dx, dy)]
            self._cur_dir = sdir
            self._cur_last_act = t

    def _close_seg(self, evs):
        """一段方向移动结束: 尝试与待定的主摆配对成背闪, 或本身成为新主摆"""
        act = [e for e in evs if abs(e[1]) >= NOISE]
        if not act:
            return
        cum = sum(abs(e[1]) for e in act)
        peak = max(abs(e[1]) for e in act)
        net = sum(e[1] for e in act)
        s, e = act[0][0], act[-1][0]

        if self._pend is not None:
            p = self._pend
            # 配对条件: 反向 + 回摆够大 + 紧接主摆 + 总时长合理
            if (net * p['net'] < 0 and
                    cum >= max(MIN_BACK_MS, p['cum'] * MIN_BACK_RATIO) and
                    s - p['e'] <= MAX_PAIR_GAP_MS and
                    e - p['s'] <= MAX_TOTAL_MS):
                full = p['evs'] + evs
                self._segments.append(full)
                direction = '右' if p['net'] > 0 else '左'
                dur = e - p['s']
                print(f'  ✔ 捕获背闪 #{len(self._segments)}: {direction}向 '
                      f'甩{p["cum"]:.0f}回{cum:.0f}, 总时长 {dur:.0f}ms', flush=True)
                self._pend = None
                if len(self._segments) >= TARGET_SEGMENTS:
                    self._finished = True
                return
            # 相邻段配对失败 (方向不对/太小/太迟), 旧主摆不可能再配对
            self._pend = None
            self._discard_count += 1

        # 闭合的段本身是否够格当主摆 (快速大幅甩动)
        if cum >= MIN_MAIN and peak >= MAIN_PEAK and (e - s) <= MAIN_DUR_MS:
            self._pend = {'evs': evs, 's': s, 'e': e, 'cum': cum, 'peak': peak, 'net': net}
        else:
            # 幅度不够 -> 丢弃 (点击/微动/慢移动)
            self._discard_count += 1

    def _setup(self):
        """连接串口 + 启动鼠标捕获"""
        # 连接 CH32V305 (同步方式: 直接 open 串口)
        try:
            import serial as pyserial
            port = SerialForwarder.find_ch32v305_port()
            if port:
                s = pyserial.Serial(port=port, baudrate=115200,
                                    bytesize=pyserial.EIGHTBITS,
                                    parity=pyserial.PARITY_NONE,
                                    stopbits=pyserial.STOPBITS_ONE,
                                    timeout=0.1, write_timeout=0)
                # 包装成简易转发对象 (仅用 _serial / is_connected)
                class _Fwd:
                    def __init__(self, ser):
                        self._serial = ser
                        self.is_connected = True
                self._serial = _Fwd(s)
                self._forward = True
                print(f'✔ 已连接 CH32V305 ({port}), 鼠标将转发到目标 PC')
            else:
                print('✘ 未找到 CH32V305, 只能本地录 (游戏视角不会动)')
        except Exception as ex:
            print(f'✘ 串口初始化失败: {ex}')

        self._t0 = time.time()
        self._monitor = RawMouseMonitor(on_event=self._on_event)
        self._monitor.start()

    def _teardown(self):
        if self._monitor:
            self._monitor.stop()
        if self._serial:
            try:
                s = getattr(self._serial, '_serial', None)
                if s and s.is_open:
                    s.close()
            except Exception:
                pass
            self._serial = None

    # ================================================================
    # 录制主流程
    # ================================================================
    def run_raw(self):
        """仅录制原始事件流 (不过滤), 供分析真实背闪特征"""
        print('=' * 60)
        print('原始数据录制模式 (不过滤)')
        self._setup()
        print('请做: 几次点击按钮微动 + 几次完整背闪 (快速甩+拉回), 左右都做')
        print('=' * 60)
        try:
            input('录制中... 完成动作后按 Enter 停止\n')
        finally:
            self._teardown()

        if len(self._all_events) == 0:
            print('未录到数据')
            sys.exit(1)
        with open(RAW_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(self._all_events, f)
        duration = self._all_events[-1][0]
        total_move = sum(abs(e[1]) + abs(e[2]) for e in self._all_events)
        print(f'已保存 {len(self._all_events)} 事件 / {duration:.0f}ms / '
              f'总移动 {total_move} counts -> {RAW_OUTPUT_PATH}')

    def run(self):
        print('=' * 60)
        print(f'背闪轨迹录制 (需 {TARGET_SEGMENTS} 次, 达到自动结束; Ctrl+C 提前结束)')
        self._setup()
        print('每次: 快速把视角甩向一侧 (像被闪躲开), 再快速拉回')
        print('      左右方向都行, 中间点击按钮/开枪都无所谓(自动过滤)')
        print('      整个动作约 0.15-0.4 秒, 动作间停顿 1-2 秒')
        print('=' * 60)
        print('等待背闪...', flush=True)
        try:
            while not self._finished:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print('\n手动停止')
        finally:
            # 强制闭合末尾段 (可能正好是一次完整动作的回摆)
            self._close_seg(self._cur) if self._cur else None
            self._teardown()

        if len(self._segments) == 0:
            print('未捕获到有效背闪动作, 退出')
            sys.exit(1)

        print(f'\n录制完成: {len(self._segments)} 段, 开始学习...')
        traj = self._learn()
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(traj, f)
        print(f'已保存: {OUTPUT_PATH}')
        print(f'  轨迹时长: {traj["t_ms"][-1]:.0f}ms, '
              f'最远转角 {max(traj["dx_cum"]):.0f} counts, 净位移 {traj["dx_cum"][-1]:.0f}')

    # ================================================================
    # 学习: 归一化为标准右背轨迹
    # ================================================================
    @staticmethod
    def _straighten(avg_c):
        """回正补偿: 把"甩开-拉回"轨迹的拉回段放大, 使最终残留视角差很小

        录制的真实动作常没完全拉回 (残留可达主摆 30%+), 放大位移倍数后
        残差更明显, 视角会偏着回不来。做法: 找到最远转角点 (累计峰值 P),
        峰值之后的回摆段按比例 k 拉伸, 使终点回到 P 的 RESIDUAL_RATIO 处
        (默认 5%, 只留少量自然差异)。回摆形状不变, 只是幅度略大。
        """
        import numpy as np
        i_peak = int(np.argmax(avg_c))
        p = float(avg_c[i_peak])
        end = float(avg_c[-1])
        # 已回正 (残差 <=5%) 或终点不低于峰值 (无回摆/回摆过头): 不动
        if p <= 0 or end >= p or end <= p * RESIDUAL_RATIO:
            return avg_c
        target = p * RESIDUAL_RATIO
        k = (target - p) / (end - p)   # end 在 (5%P, P) 内 -> k > 1, 拉伸回摆段
        out = avg_c.copy()
        out[i_peak + 1:] = p + k * (avg_c[i_peak + 1:] - p)
        return out

    def _learn(self):
        """C(t) 累计位移插值法: 各段时长归一, 累计曲线对齐平均, 再差分

        直接对事件位移插值会把 8ms 事件间隔搬到 3ms 采样网格,
        位移被放大 ~2.8 倍; 对累计位移 C(t) 插值则精确守恒。
        """
        import numpy as np

        # 方向统一为"右背" (主摆 dx>0), 裁掉首尾噪声; dy 保留形状 (不镜像)
        right = []
        for seg in self._segments:
            act_idx = [i for i, e in enumerate(seg) if abs(e[1]) >= NOISE]
            if not act_idx:
                continue
            seg = seg[act_idx[0]:act_idx[-1] + 1]
            if sum(e[1] for e in seg) < 0:
                seg = [(t, -dx, dy) for t, dx, dy in seg]
            ts = np.array([e[0] - seg[0][0] for e in seg], dtype=float)
            Cx = np.cumsum([e[1] for e in seg])
            Cy = np.cumsum([e[2] for e in seg])
            right.append((ts, Cx, Cy))

        durations = [ts[-1] for ts, _, _ in right]
        target_dur = float(np.median(durations))
        N = 100
        t_ms = np.linspace(0, target_dur, N)
        # 各段累计曲线插值到公共网格后平均, 差分得每采样点位移
        cx_all = np.array([np.interp(t_ms, ts, Cx) for ts, Cx, _ in right])
        cy_all = np.array([np.interp(t_ms, ts, Cy) for ts, _, Cy in right])
        avg_cx = self._straighten(cx_all.mean(axis=0))
        avg_cy = cy_all.mean(axis=0)
        dx = np.diff(avg_cx, prepend=0.0)
        dy = np.diff(avg_cy, prepend=0.0)

        return {
            't_ms': [round(float(x), 1) for x in t_ms],
            'dx': [round(float(x), 1) for x in dx],
            'dy': [round(float(x), 1) for x in dy],
            'dx_cum': [round(float(x), 1) for x in avg_cx],  # 累计曲线 (校验用)
        }


if __name__ == '__main__':
    rec = BackflashRecorder()
    if len(sys.argv) > 1 and sys.argv[1] == '--raw':
        rec.run_raw()
    else:
        rec.run()
