import cv2
import gym
import matplotlib.pyplot as plt

import numpy as np
from .env_module import EnvModule
from tools.config import Configurable, as_builder, merge_inputs


@as_builder
class Observer(Configurable, EnvModule):
    def __init__(self, cfg=None):
        super(Observer, self).__init__()

    @property
    def observation_space(self):
        raise NotImplementedError(f"{self.__class__}'s observation space is not implemented.")

    def get_obs(self, **kwargs):
        raise NotImplementedError

    def render_array(self, **kwargs):
        raise NotImplementedError

    def render_plt(self, img):
        plt.imshow(np.uint8(img[:,:,:3]))
        plt.show()

    def render_human(self, img):
        cv2.imshow('x', img)
        cv2.waitKey(1)

    def render(self, mode='plt', **kwargs):
        assert mode in ['plt', 'human', 'array', 'rgb_array']
        img = self.render_array(**kwargs)
        if mode == 'array':
            return img
        elif mode == 'rgb_array':
            # make sure that the output is a rgb array ..
            return np.uint8(img[:, :, :3]) 
        elif mode == 'human':
            self.render_human(img)
        else:
            self.render_plt(img)


class TaichiObserver(Observer):
    def __init__(self, cfg=None, center=(0.5, 0.2, 0.5), theta=np.pi/4, phi=0., radius=3., primitive=1):
        super(TaichiObserver, self).__init__()
        #from plb.engine.renderer.renderer import Renderer
        self.renderer = None

    def render_array(self, **kwargs):
        from tools.utils import lookat # optimize import speed ..

        cfg = merge_inputs(self._cfg, **kwargs)
        self.renderer.setRT(*lookat(cfg.center, cfg.theta, cfg.phi, cfg.radius))
        return self._render_fn(cfg)

    def set_initial_camera(self):
        from tools.utils import lookat # optimize import speed ..
        self.renderer.setRT(*lookat(
            self._cfg.center, self._cfg.theta, self._cfg.phi, self._cfg.radius))

    def set_env(self, env):
        super(TaichiObserver, self).set_env(env)
        self.renderer = self.env.renderer
        self.set_initial_camera()


class TaichiRGBDObserver(TaichiObserver):
    def __init__(self, cfg=None, spp=1, shape=1):
        super(TaichiRGBDObserver, self).__init__()

    def get_obs(self):
        return None

    def _render_fn(self, cfg):
        return self.env._render(spp=cfg.spp, primitive=cfg.primitive, shape=cfg.shape)


class ParticleObserver(TaichiRGBDObserver):
    # render in camera, but get the observation from the image ..
    def __init__(self, cfg=None, n_particles=100):
        super(ParticleObserver, self).__init__()

    @property
    def observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (self._cfg.n_particles * 5 + 9 + 9,))

    def get_obs(self):
        from .diffenv import DifferentiablePhysicsEnv
        self.env: DifferentiablePhysicsEnv

        state = self.env._taichi_state()
        pos = state['pos'][:self._cfg.n_particles].detach().cpu().numpy()
        dist = state['dist'][:self._cfg.n_particles].detach().cpu().numpy()
        tool = state['tool'].detach().cpu().numpy()
        
        import transforms3d
        tool = np.concatenate(sum([[i[:3], transforms3d.quaternions.quat2mat(i[3:])[:, :2].reshape(-1)] for i in tool], []))

        return np.concatenate((pos.reshape(-1), dist.reshape(-1), tool.reshape(-1)))
