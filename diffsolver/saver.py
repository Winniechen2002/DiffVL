import os
import torch
import copy
import numpy as np
from diffsolver.program.types import OBSDict

from diffsolver.utils import MultiToolEnv, WorldState
from .config import SaverConfig
from tools.utils import logger
from omegaconf.omegaconf import OmegaConf
from typing import Optional, cast, Dict
from .program.types import Constraint, SceneSpec, OBSDict
from .program.scenes import SceneTuple
from .utils import MultiToolEnv, WorldState
from numpy.typing import NDArray
import tqdm


class ForwardCallback:
    def on_frame(self, frame: int, obs: OBSDict):
        pass

    def on_start(self, engine, env: MultiToolEnv, state: WorldState, actions: torch.Tensor, requir_grad: bool):
        from .engine import Engine
        assert not requir_grad, 'requir_grad is not supported yet'
        self.engine: Engine = engine
        self.env, self.state, self.actions, self.requires_grad = env, state, actions, requir_grad

    def on_finish(self):
        pass


class RenderCallback(ForwardCallback):
    def __init__(self, render_mode: str):
        self.images = []
        self.ending = None
        self.render_mode = render_mode
        assert render_mode in ['rgb_array', 'human']

    def render(self, **kwargs):
        if self.render_mode == 'rgb_array':
            out = self.env.render('rgb_array', **kwargs)
        else:
            from llm.tiny import Scene
            self.engine.gui.load_scene(Scene(self.env))
            out = self.engine.gui.capture()
            # raise NotImplementedError('human mode is not supported yet')
        return cast(NDArray[np.uint], out)

    def on_frame(self, frame: int, obs: OBSDict):
        self.images.append(self.render())

    def on_finish(self):
        self.ending = self.render(primitive=0)



class StateSaver(ForwardCallback):
    def __init__(self) -> None:
        self.obs = []
        self.states = []

    def on_start(self, engine, env: MultiToolEnv, state: WorldState, actions: torch.Tensor, requir_grad: bool):
        super().on_start(engine, env, state, actions, requir_grad)
        self.states.append(env.get_state())
        self.obs.append(env.get_obs())

    def on_frame(self, frame: int, obs: OBSDict):
        self.states.append(self.env.get_state())
        self.obs.append(obs)


class Saver:
    def __init__(self, engine, config: SaverConfig) -> None:
        from .engine import Engine
        self.config = config
        self.engine: Engine = engine

    def __enter__(self):
        log_cfg = self.config.logger
        kwargs = {}

        if self.config.use_wandb:
            kwargs = {'project': os.environ.get("WANDB_PROJECT_NAME", 'diffsolver'), 'group': self.config.wandb_group, 'name': self.config.wandb_name, 'config': self.engine.config}
            log_cfg.format_strs += ['wandb']

        assert len(self.config.path) != 0
        self.dir = self.config.path
        if not logger.is_configured():
            logger.configure(
                self.config.path, format_strs=log_cfg.format_strs, date=log_cfg.date, **kwargs
            )
            #self.dir = logger.get_dir()
            assert logger.get_dir() == self.config.path
            self.prefix = ''
        else:
            self.prefix = os.path.relpath(self.config.path, logger.get_dir()) + '/'
            os.makedirs(self.config.path, exist_ok=True)

        self.scene = None
        self.actions = None
        self.best_actions = None
        self.best_loss = np.inf
        self.idx = 0

        return self

    def save_config(self, config, filename='config.yml'):
        with open(os.path.join(self.dir, filename), 'w') as f:
            f.write(OmegaConf.to_yaml(config))

    def write_file(self, output: str, filename: str):
        with open(os.path.join(self.dir, filename), 'w') as f:
            f.write(output)

    def write_video(self, output, filename: str):
        logger.animate(output, self.prefix + filename, fps = 10)

    def visualizer(self, scene: SceneSpec, actions: NDArray[np.float32], save_trajs: Optional[str] = None):
        renderer_callback = RenderCallback(self.config.render_mode)
        callbacks = (renderer_callback,)

        state_saver = None
        if save_trajs is not None:
            state_saver = StateSaver()
            callbacks += (state_saver,)

        _actions = torch.tensor(actions, dtype=torch.float32)
        self.engine.forward(self.engine.env, scene.state, _actions, requires_grad=False, callbacks=callbacks)

        if state_saver is not None:
            assert save_trajs is not None
            torch.save(
                {
                    'states': state_saver.states,
                    'obs': state_saver.obs,
                    'actions': actions,
                    'names': scene.state_tuple.names
                },
                os.path.join(self.dir, save_trajs)
            )
        return renderer_callback.images, renderer_callback.ending

    def __exit__(self, exc_type, exc_value, traceback):
        """finalize"""
        if not exc_type and self.best_actions is not None and self.scene is not None:
            """
            store the best trajectories
            """

            images, ending_image = self.visualizer(
                self.scene, self.best_actions, save_trajs='trajs.pt')
            logger.animate(images, self.prefix + 'best.mp4', use_html=False)
            logger.savefig(self.prefix + 'ending.png', ending_image)

            # plot the curves
            # import pandas as pd
            # database = pd.read_csv(os.path.join(self.dir, 'progress.csv'))
            # ax = database.plot()
            # ax.legend([col.replace('>>', ',') for col in database.columns])
            # logger.savefig(self.prefix + 'curve.png')

    def save_task_info(self, env: MultiToolEnv, stage: SceneTuple):
        self.write_file(str(env._cfg), 'env_config.yaml')
        assert env.tool_cur is not None
        if env.tool_cur._cfg is not None:
            self.write_file(str(env.tool_cur._cfg), 'tool_config.yaml')
        logger.savefig(self.prefix + 'start.png', env.render('rgb_array'))
        goal_image = stage.get_goal_image()
        if goal_image is not None:
            logger.savefig(self.prefix + 'goal.png', goal_image)

    def step(self, scene: SceneSpec, actions, constraints: Constraint, trange: Optional[tqdm.tqdm] = None, infos: Optional[Dict] = None):
        self.scene = scene
        loss = constraints.loss.item()

        self.actions = actions
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_actions = copy.deepcopy(actions)

        if infos is not None and len(infos) > 0:
            for k, v in infos.items():
                for j, v2 in v.items():
                    logger.logkv(self.prefix + f"{k}.{j}", v2)

        logger.logkv(self.prefix + 'loss', loss)
        for c in constraints.tolist():
            logger.logkv(self.prefix + c['code'], float(c['loss']))
        logger.dumpkvs()

        self.idx += 1
        if self.idx % self.config.render_interval == 0:
            images, ending = self.visualizer(scene, actions)
            logger.animate(images, self.prefix + f'iter_{self.idx}.mp4', use_html=False)
            logger.savefig(self.prefix + f'ending_{self.idx}.png', ending)

        if trange is not None:
            trange.set_description(
                f'loss: {loss:.4f},  best: {self.best_loss:.4f}')

        return loss, self.best_loss
