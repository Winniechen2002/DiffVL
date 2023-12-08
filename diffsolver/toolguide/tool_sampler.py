import torch
import termcolor
import logging
import numpy as np
from dataclasses import dataclass, field
from omegaconf import DictConfig
from typing import List, Mapping,  Type, Optional, cast
import tqdm
from numpy.typing import NDArray
from ..program.types import ToolSpace
from ..program.types.tool_space import EmptySpace
from ..program.types.tool_space import Gripper, DoublePushers, Pusher

from .tool_progs.equations import get_tool_eq_fn
from .tool_progs.constraints import get_tool_cons
from ..program.types import SceneSpec
from ..program.scenes import SceneTuple
from ..utils import MultiToolEnv
from ..config import ToolSamplerConfig
from .motion_planning import MotionPlanner

    
TOOL_SPACES: Mapping[str, Type[ToolSpace]] = {
    'Gripper': Gripper,
    'DoublePushers': DoublePushers,
    'Pusher': Pusher,
}

STAGE_ID = 0

class ToolSampler:
    def __init__(self, env: MultiToolEnv, config: ToolSamplerConfig) -> None:
        self.env = env

        self.config = config
        self.equations = [get_tool_eq_fn(eq) for eq in self.config.equations]
        self.constraints = [get_tool_cons(cons) for cons in self.config.constraints]
        self.mp_mp4 = None

    def select_tool(self, scene_tuple: SceneTuple):
        tool_name = scene_tuple.state.tool_name
        
        sampler = self.config.sampler
        if sampler == 'default':
            sampler = tool_name
        return TOOL_SPACES.get(sampler, EmptySpace)(scene_tuple.state.tool_cfg, self.config.use_tips)

    def compute_sdf(self, sample: torch.Tensor):
        f = 0
        state = self.env.simulator.states[f]
        assert self.env.tool_cur is not None
        pos, rot = self.env.tool_cur.forward_kinematics(sample)
        state.body_pos.upload(pos.detach().cpu().numpy())
        state.body_rot.upload(rot.detach().cpu().numpy())
        out = self.env.simulator.get_dists(f, grad=None, device=pos.device)
        assert isinstance(out, torch.Tensor)
        return out

    def solve(self, _scene_tuple: SceneTuple, tool_space: ToolSpace):
        old_scene_state = _scene_tuple.state
        scene_tuple = SceneTuple(_scene_tuple.state, _scene_tuple.names, _scene_tuple.goal, _scene_tuple.config, _scene_tuple.goal_names)
        assert scene_tuple.state is not old_scene_state,  "Scene state is not copied."
        scene = SceneSpec.from_scene_tuple(self.env, scene_tuple, tool_space, requires_grad=False)


        assert self.env.tool_cur is not None, "Tool is not set."
        for fn in self.equations:
            index, value = fn(scene)

            tool_space.set_values(index, value)


        samples = torch.tensor(tool_space.sample(self.config.n_samples), dtype=torch.float32)
        best_loss = float(1e9)
        best_results = torch.tensor(scene_tuple.state.qpos, dtype=torch.float32)

        it = tqdm.tqdm(samples, desc='Sampling tool poses', total=len(samples))

        solutions: List[torch.Tensor] = []

        valid = 0
        count = 0

        for sample in it:
            count += 1
            dists = self.compute_sdf(sample)

            # print(sample,dists)
            assert dists is not None
            assert isinstance(dists, torch.Tensor)
            scene.obs[0]['dist'] = dists
            scene.obs[0]['qpos'] = sample

            loss = 0 
            reject = False

            for prog in self.constraints:
                weights, cond =  prog(scene)
                if cond is None:
                    loss += weights
                    continue

                if not cond:
                    reject = True
                    break

                    
            valid = valid + (not reject)
            it.set_description(f"Sampling tool poses (loss: {loss:.3f}, best: {best_loss:.3f}) valid {valid}/{count} = {valid/count:.3f}")
            if reject:
                continue

            if loss < best_loss:
                best_loss = loss
                best_results = sample

            if len(solutions) < self.config.n_sol:
                solutions.append(sample)

            if not self.config.optimize_loss and len(solutions) >= self.config.n_sol:
                break

        return float(best_loss), best_results, solutions

    def update_scene_tuple(self, scene: SceneSpec, prefix:str=''):
        _, results, sols = self.solve(scene.state_tuple, scene.tool_space)
        if len(sols) > 0:
            from tools.utils import logger
            import cv2, os
            images = []
            for q in sols:
                #img = np.concatenate([img, goal_image], axis=1)
                self.compute_sdf(q)
                img = self.env.render('rgb_array')
                images.append(img)

            images = torch.tensor(np.array(images), dtype=torch.uint8).permute(0, 3, 1, 2)

            from torchvision.utils import make_grid
            grid = make_grid(images, nrow=5)
            ndarr = grid.clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            cv2.imwrite(os.path.join(logger.get_dir(), 'sols.png'), ndarr[:,:,::-1])
        else:
            raise RuntimeError("No tool solutions found.")

        old_scene_state = scene.state_tuple.state

        def detach(x) -> NDArray[np.float64]:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        if self.config.motion_planner.max_iter > 0:
            path = self.motion_planning(
                scene.state_tuple, 
                scene.tool_space, 
                detach(old_scene_state.qpos), 
                detach(results)
            )
            self.render_tool_paths(scene.state_tuple, path, prefix)


        print(termcolor.colored(f"qpos {results}.", "red"))
        scene.state_tuple.state = old_scene_state.switch_tools(tool_name=old_scene_state.tool_name, qpos=results.detach().cpu().numpy())
        return sols

    def render_tool_paths(self, state_tuple: SceneTuple, qpos: List[NDArray[np.float64]], prefix: str):
        if len(qpos) == 0:
            logging.warn(termcolor.colored("No path found.", "red"))
            return
        from ..saver import logger
        self.env.set_state(state_tuple.state)
        images = []
        for q in qpos:
            self.compute_sdf(torch.tensor(q, dtype=torch.float32).to(self.env.device))
            images.append(self.env.render('rgb_array'))

        logger.torch_save(qpos, prefix + 'mp.th')
        logger.animate(images, prefix + 'mp.mp4', fps=10)
        self.mp_mp4 = images
        
    def motion_planning(
        self, 
        scene_tuple: SceneTuple, 
        tool_space: ToolSpace, 
        start_qpos: NDArray[np.float64],
        goal_qpos: NDArray[np.float64]
    ):
        self.env.set_state(scene_tuple.state)
        mp_config = self.config.motion_planner
        state_sampler = tool_space.get_state_sampler(start_qpos, goal_qpos)


        def collision_checker(qpos: NDArray[np.float64]):
            dists = self.compute_sdf(
                torch.tensor(qpos, dtype=torch.float32).to(self.env.device))
            assert dists is not None
            assert isinstance(dists, torch.Tensor)
            return dists.min().item() < mp_config.tolerance

        motion_planner = MotionPlanner(state_sampler, collision_checker, mp_config)
        return motion_planner(start=start_qpos, goal=goal_qpos)