"""
config.py - 持久化配置管理

保存/加载用户参数到 JSON 文件，重启后自动恢复。

可靠性设计 (防方案丢失事故):
  - 深拷贝默认配置: 实例之间、实例与 DEFAULT_CONFIG 之间互不污染
  - 原子写: 先写临时文件再 os.replace, 任何时刻读到的都是完整 JSON
    (直接 open('w') 会先截断文件, 并发/崩溃时读到半截 -> 解析失败)
  - load 失败保护: 文件存在但解析失败 -> 备份为 .bad-<时间> 且禁止本次
    自动保存覆盖 (避免"解析失败->静默用默认->写盘把用户数据冲掉"的事故链)
  - 单实例锁: 第二个后端实例禁止写 config (多实例交替写会造成损坏)
"""

import copy
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_CONFIG = {
    'trajectory': {
        'enabled': False,
        'smooth_factor': 0.35,
        'max_step_px': 10,
        'min_confidence': 0.30,
        'target_offset_x': 0,
        'target_offset_y': 0,
        'jitter_amount': 0.15,
        'target_priority': -1,
        'prediction_ticks': 3,
        'fov_radius': 300,
        'fov_pull_enabled': False,
        'fov_pull_speed': 2.0,
        'ai_scale': 1.0,
        'trigger_enabled': False,
        'trigger_threshold': 5,
        'invert_ai_x': False,
        'invert_ai_y': False,
        'kp': 0.35,
        'ki': 0.02,
        'kd': 0.10,
        'integral_limit': 100.0,
        'max_steps_per_frame': 30,
        'settle_deadzone': 8.0,
        'unsettle_hysteresis': 20.0,
        'y_scale': 0.2,
    },
    'video': {
        'show_video': True,
        'capture_width': 1920,
        'capture_height': 1080,
        'capture_fps': 240,
    },
    'mouse': {
        'source': 'raw',   # 鼠标源: 'raw' = raw input (FPS 兼容), 'pynput' = 光标模式
        # 注意: 物理鼠标倍率 (scale) 为废案, 已移除, 不再生效
    },
    'screen': {
        'target_w': 2560,
        'target_h': 1440,
    },
    'model': {
        'last_path': '',
    },
    'lock': {
        'auto_lock': True,  # 采集卡+模型加载完成后自动进入锁定模式
    },
    'presets': {
        'current': 1,       # 当前选中的方案
        '1': {'name': '无畏契约', 'params': {}},
        '2': {'name': '三角洲行动', 'params': {}},
        '3': {'name': '', 'params': {}},
        '4': {'name': '', 'params': {}},
        '5': {'name': '', 'params': {}},
    },
}


def _pid_alive(pid: int) -> bool:
    """检查进程是否存活 (Windows)"""
    try:
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        # 退出码 STILL_ACTIVE (259) 表示存活
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259
    except Exception:
        return True  # 无法判断时保守认为存活


class Config:
    """持久化配置"""

    def __init__(self, config_dir: str = None, allow_write: bool = True):
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(__file__), '..')
        self._path = os.path.join(config_dir, 'config.json')
        # 深拷贝: 嵌套 dict 与 DEFAULT_CONFIG 完全隔离 (浅拷贝会全局污染)
        self._data = copy.deepcopy(DEFAULT_CONFIG)
        self._loaded_ok = True       # 文件正常解析
        self._write_allowed = allow_write
        self.load()

    @property
    def path(self) -> str:
        return self._path

    def get(self, *keys, default=None):
        """获取配置值"""
        d = self._data
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k)
                if d is None:
                    return default
            else:
                return default
        return d if d is not None else default

    def set(self, *args):
        """
        设置配置值

        用法:
          config.set('trajectory', 'smooth_factor', 0.5)
          config.set('trajectory', 'enabled', True)
        """
        if len(args) < 2:
            return
        *keys, value = args
        d = self._data
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
        self.save()

    def update_trajectory_config(self, **kwargs):
        """批量更新轨迹配置"""
        for k, v in kwargs.items():
            if k in self._data['trajectory']:
                self._data['trajectory'][k] = v
        self.save()

    def load(self):
        """从文件加载

        文件存在但解析失败时: 备份坏文件 + 禁止继续自动保存,
        避免"解析失败 -> 静默用默认 -> 写盘冲掉用户数据"的事故。
        """
        try:
            if os.path.exists(self._path):
                with open(self._path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if not isinstance(loaded, dict):
                    raise ValueError('config root is not a dict')
                # 合并到默认配置 (保留默认值, 覆盖已保存的)
                self._deep_merge(self._data, loaded)
                self._loaded_ok = True
                logger.info(f"Config loaded: {self._path}")
            else:
                self._loaded_ok = True
                logger.info("No config file, using defaults")
        except Exception as e:
            self._loaded_ok = False
            self._write_allowed = False
            # 备份坏文件, 保留恢复证据 (绝不覆盖它)
            bad_path = self._path + f'.bad-{time.strftime("%Y%m%d-%H%M%S")}'
            try:
                os.replace(self._path, bad_path)
                logger.error(f"Config parse failed ({e}); "
                             f"备份为 {bad_path}, 本次运行禁止写配置")
            except Exception as e2:
                logger.error(f"Config parse failed ({e}); 备份失败: {e2}; "
                             f"本次运行禁止写配置")

    def save(self):
        """保存到文件 (原子写)"""
        if not self._write_allowed:
            logger.warning("Config 写入已被禁用 (load 失败保护), 跳过保存")
            return
        try:
            tmp_path = self._path + '.tmp'
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)  # 原子替换: 读方永远看到完整文件
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")

    def _deep_merge(self, base, override):
        """递归合并字典"""
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v


# ================================================================
# 单实例锁: 第二个后端实例禁止写 config
# (两个实例交替写同一文件 -> 一个截断时另一个读到半截 -> 解析失败 -> 方案丢失)
# ================================================================
_LOCK_PATH = None


def acquire_backend_lock(config_dir: str = None) -> bool:
    """尝试获取后端单实例锁

    Returns:
        True  = 本进程是唯一实例 (可写 config)
        False = 已有实例在运行 (本进程禁止写 config, 但功能仍可用)
    """
    global _LOCK_PATH
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(__file__), '..')
    _LOCK_PATH = os.path.join(config_dir, 'backend.lock')
    try:
        if os.path.exists(_LOCK_PATH):
            with open(_LOCK_PATH, 'r') as f:
                pid = int(f.read().strip() or 0)
            if pid and pid != os.getpid() and _pid_alive(pid):
                logger.warning(
                    f"检测到另一个后端实例 (PID {pid}) 正在运行; "
                    f"本实例禁止写入 config.json, 避免配置互相覆盖损坏")
                return False
        with open(_LOCK_PATH, 'w') as f:
            f.write(str(os.getpid()))
        return True
    except Exception as e:
        logger.warning(f"Backend lock error: {e}")
        return True  # 锁机制异常时放行, 不阻挡正常使用


def release_backend_lock():
    """退出时释放锁"""
    global _LOCK_PATH
    if not _LOCK_PATH:
        return
    try:
        if not os.path.exists(_LOCK_PATH):
            return
        with open(_LOCK_PATH, 'r') as f:
            owner = int(f.read().strip() or 0)
        # 注意: 必须在文件句柄关闭后再 remove (Windows 不允许删除打开中的文件)
        if owner == os.getpid():
            os.remove(_LOCK_PATH)
    except Exception:
        pass
