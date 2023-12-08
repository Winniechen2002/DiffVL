import tqdm
import torch
import random
from .config import DefaultConfig
from .utils import MultiToolEnv, ConfigType, WorldState
from .program.types import SceneSpec, OBSDict
from .optimizer import Optimizer, FrankWolfe
from .saver import Saver, ForwardCallback
from .evaluator.evaluator import Evaluator

from .program.prog_registry import build_prog, SCENE_COND
from .program import progs # noqa: F401 

from typing import Sequence, List, Optional, cast

from .program.scenes import load_scene_with_envs
from .toolguide import ToolSampler
from omegaconf import DictConfig, OmegaConf

from tools.utils import logger
import numpy as np

from .toolguide.prompts.get_prompts import answer
from .toolguide.parse_strings import parse2yaml

class Engine:
    def __init__(
        self,
        env: MultiToolEnv,
        config: ConfigType[DefaultConfig],
    ):
        super().__init__()
        self.config = config

        if self.config.seed is not None:
            random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)

        self.env = env

        if self.config.saver.render_mode == 'human':
            from .utils.renderer import CameraGUI
            if self.config.saver.use_lookat:
                self.gui = CameraGUI()
                self.gui.lookat(self.config.saver.lookat_config)
            else:
                self.gui = CameraGUI(camera_config = self.config.saver.camera_config)

    def forward(self, 
        env: MultiToolEnv, state: WorldState, actions: torch.Tensor, requires_grad=True,
        callbacks: Sequence[ForwardCallback] = (),
        obs: List[OBSDict] | None = None,
        evaluator: Optional[Evaluator] = None,
    ):
        env.set_state(state, requires_grad=requires_grad)

        for callback in callbacks:
            callback.on_start(self, env, state, actions, requires_grad)
        obs = [] if obs is None else [o for o in obs] # copy the list
        obs.append(env.get_obs())

        if evaluator is not None:
            evaluator.reset(obs[-1])

        for i in range(len(actions)):
            env.step(actions[i])
            obs.append(env.get_obs())

            for callback in callbacks:
                callback.on_frame(i, obs[-1])

            if evaluator is not None:
                with torch.no_grad():
                    evaluator.onstep(obs[-1], actions[i], locals())

        for callback in callbacks:
            callback.on_finish()
        
        if evaluator is not None:
            with torch.no_grad():
                return obs, evaluator.onfinish()
        return obs, {}

        
    def get_init_actions(self, 
        actions: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None, 
    ):
        init_action = None
        if horizon is not None:
            init_action = torch.zeros(
                (horizon, self.env.action_space.shape[-1]), dtype=torch.float32)
        if actions is not None:
            if init_action is not None:
                l = min(len(actions), len(init_action))
                init_action[:l] = actions[:l]
            else:
                init_action = actions
        assert init_action is not None, "horizon or actions must be provided"
        return init_action



    def train_diffphys(
        self, 
        scene: SceneSpec, program: SCENE_COND, saver: Saver,
        actions: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None, 
        prev_obs: Optional[List[OBSDict]] = None,
    ):
        trainer = self.config.trainer
        OPTIM_CLS = Optimizer if not self.config.optim.enable_constraints else FrankWolfe 

        optimizer = OPTIM_CLS(self.env, self.get_init_actions(actions, horizon), self.config.optim)
        trange = tqdm.trange(trainer.nsteps)

        evaluator = Evaluator(scene, self.config.evaluator)


        for idx in trange: 
            with optimizer as o:

                observations, infos = self.forward(self.env, scene.state, optimizer.parameters, requires_grad=True, obs=prev_obs, evaluator=evaluator)
                scene.obs = observations
                constraint = program(scene)
                if idx == 0:
                    logger.log("Translated program: ", constraint.synchronize())
                o.step(constraint)

            saver.step(scene, optimizer.parameters.data, constraint, trange, infos=infos)


    def main(self) -> str:
        with Saver(self, self.config.saver) as saver:
            saver.save_config(self.config, 'config.yaml')

            cfg = self.config
            if cfg.tool_sampler.use_lang and cfg.tool_sampler.n_samples > 0:
                if cfg.tool_sampler.use_default:
                    new_config = {
                        'sampler': {
                            'n_samples': 0,
                        },
                        'scene': {
                            'Tool':{
                                'tool_name': 'Gripper',
                                'qpos': [0.5, 0.15, 0.5, 0., 0, 0., 0.7],
                                "size": (0.03, 0.08, 0.03),
                            }
                        }
                    }
                else:
                    if not cfg.tool_sampler.use_code:
                        if cfg.tool_sampler.use_cpdeform:
                            result = """- locate(get('all'), 0.3)\n- cpdeform('all')"""
                        else:
                            assert len(cfg.tool_sampler.lang) > 0, "language prompt for tool sampling must be provided"
                            result = answer(cfg.tool_sampler.lang)
                            saver.write_file(result, 'tool_lang.txt')
                    else:
                        assert cfg.tool_sampler.code is not None, "code for tool sampling must be provided"
                        result = str(cfg.tool_sampler.code)

                    new_config = parse2yaml(result)
                    if 'scene' not in new_config:
                        new_config['scene'] = {}
                    logger.write_file('tool_cfg.yml', OmegaConf.to_yaml(new_config))

                cast(DictConfig, cfg).merge_with(
                    {'tool_sampler': new_config['sampler'], 'scene': new_config['scene']}
                )
            self.config = cfg

            scene_tuple = load_scene_with_envs(self.env, cfg.scene)
            self.tool_sampler = ToolSampler(self.env, cfg.tool_sampler)

            # create scene
            scene = SceneSpec.from_scene_tuple(self.env, scene_tuple, self.tool_sampler.select_tool(scene_tuple), sample = self.config.sample)

            # 1. sample tools 
            if cfg.tool_sampler.n_samples > 0:
                self.tool_sampler.update_scene_tuple(scene, prefix=saver.prefix)
                # saver.write_file(str(scene.state.qpos.tolist()), 'qpos.txt')
                # if self.tool_sampler.mp_mp4:
                #     saver.write_video(self.tool_sampler.mp_mp4, 'mp.mp4')


            self.env.set_state(scene.state)
            assert len(self.env.action_space.shape) == 1
            saver.save_task_info(self.env, scene.state_tuple)

            # 2. run diffphys
            if cfg.run_solver:
                prog = cfg.prog
                if prog.max_retry > 0:
                    from diffsolver.program.clause.translator import translate_program_from_scene


                    scene.obs = [self.env.get_obs()]
                    code = translate_program_from_scene(prog.lang, scene_tuple, cfg.prog.translator, prog.max_retry, scene=scene, tool_lang=cfg.tool_sampler.lang)

                    if code is None or code.startswith('HIS'):
                        logger.log("Failed to translate program from the language")
                        if code is not None:
                            saver.write_file(code, 'phys_lang.txt')
                        return saver.dir
                    
                    logger.log("Translated program: ", code)
                    prog.code = code
                
                task = build_prog(prog, cfg.stages)
                saver.save_config(self.config, 'new_config.yaml')
                self.train_diffphys(scene, task['prog'], saver, horizon=task['horizon'])
            else:
                saver.save_config(self.config, 'new_config.yaml')

        return saver.dir