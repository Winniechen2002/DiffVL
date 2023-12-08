from tools.config import CN

class EnvModule:
    def __init__(self):
        from .diffenv import DifferentiablePhysicsEnv
        self.env: DifferentiablePhysicsEnv = None

    def _update_cfg(self, cfg):
        return cfg

    def update_cfg(self, simulator_cfg: CN):
        simulator_cfg.defrost()
        simulator_cfg = self._update_cfg(simulator_cfg)
        simulator_cfg.freeze()
        return simulator_cfg

    def set_env(self, env):
        self.env = env



