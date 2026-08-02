"""
config.py - 持久化配置管理

保存/加载用户参数到 JSON 文件，重启后自动恢复。
"""

import json
import logging
import os

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
    'screen': {
        'target_w': 2560,
        'target_h': 1440,
    },
    'model': {
        'last_path': '',
    },
}


class Config:
    """持久化配置"""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(__file__), '..')
        self._path = os.path.join(config_dir, 'config.json')
        self._data = DEFAULT_CONFIG.copy()
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
        """从文件加载"""
        try:
            if os.path.exists(self._path):
                with open(self._path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                # 合并到默认配置 (保留默认值, 覆盖已保存的)
                self._deep_merge(self._data, loaded)
                logger.info(f"Config loaded: {self._path}")
            else:
                logger.info(f"No config file, using defaults")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    def save(self):
        """保存到文件"""
        try:
            with open(self._path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")

    def _deep_merge(self, base, override):
        """递归合并字典"""
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v