import argparse
import torch
import os
import tqdm
from typing import cast, List, Any
import numpy as np
from omegaconf import OmegaConf, DictConfig
from tools.utils import animate
from dataclasses import dataclass, field


from diffsolver.rl_baselines.rl_envs.gym_env import MultiToolEnv, CN, CameraGUI



@dataclass
class LookatConfig:
    center: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.2])
    theta: float = 0.
    phi: float = np.pi/4
    radius: float = 2.5
    zeta: float = np.pi/2


@dataclass
class Config:
    path: str
    taskname: str
    mode: str = 'ours'
    multistage: bool = False
    output: str = "examples/output/images/${mode}_${taskname}"

    ray_tracing: int = 1024
    lookat: LookatConfig = field(default_factory=LookatConfig)
    frames: None|List[int] = None
    inherit: str|None = None

    threshold: float = 0.5
    reverse: bool = False

    goal_transparency: float = 1.

    color: Any = None
    tool_pos: Any = None
    set_tool_pos: bool = False
    use_knife: bool = False
    render_goal: bool = True

    stage_tools: Any = field(default_factory=lambda: [])
    render_stages: List[int] = field(default_factory=lambda: [])

    render_mp: bool = False


def get_inherit(cfg: DictConfig, base_path=None) -> DictConfig:
    if not hasattr(cfg, 'inherit') or cfg.inherit is None:
        return cfg
    else:
        if base_path is not None:
            inherit = os.path.join(base_path, cfg.inherit)
        else:
            inherit = cfg.inherit
        p = OmegaConf.load(inherit + '.yml')
        assert isinstance(p, DictConfig)
        out = OmegaConf.merge(get_inherit(p), cfg)
        assert isinstance(out, DictConfig)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default=None)
    args, unknown = parser.parse_known_args()

    default_conf: DictConfig = OmegaConf.structured(Config)
    if args.config is not None:
        p = OmegaConf.load(args.config)
        assert isinstance(p, DictConfig)
        p = get_inherit(p, base_path=os.path.dirname(args.config))
        default_conf.merge_with(p)

    input_cfg = OmegaConf.from_dotlist(unknown)
    default_conf.merge_with(input_cfg)
    OmegaConf.resolve(default_conf)

    cfg = cast(Config, default_conf)
    print(OmegaConf.to_yaml(cfg))

    gui = CameraGUI(offscreen=True, ray_tracing=256)

    from tools.utils import lookat # optimize import speed ..
    import transforms3d

    lookat = cfg.lookat
    R = transforms3d.euler.euler2mat(lookat.theta, lookat.phi, lookat.zeta, 'sxyz')
    b = np.array([lookat.radius, 0., 0.], dtype=float)
    back = R[0:3, 0:3].dot(b)
    gui.setup_camera(lookat.center - back, transforms3d.quaternions.mat2quat(R))


    env = MultiToolEnv(sim_cfg=CN(dict(max_steps=100)))



    images = []

    os.makedirs(cfg.output, exist_ok=True)



    def render_state(state, remove_tool=False, set_tool_pos=False, tool_type=None, use_knife=cfg.use_knife, qpos=None, obj_ids=None, objects=()):
        gui.use_knife = use_knife

        if remove_tool:
            import copy
            state = copy.deepcopy(state)
            state.qpos[0] = -100
            state.rigid_bodies = None
        print(state.qpos)

        if obj_ids is not None:
            state.X[state.ids != obj_ids] = 100
        
        if tool_type is not None:
            state = state.switch_tools(tool_type)
        print(set_tool_pos)
        if set_tool_pos and qpos is not None:
            state.qpos = np.array(qpos)

        state.qpos[..., 1] -= 0.03
        state.rigid_bodies = None

        env.set_state(state)
        gui.reset(env)
        for i in objects:
            gui._add_pcd(**i)
        img = gui.capture()
        for k, v in gui._render_mesh.items():
            gui._scene.remove_actor(v['pcd'])
        gui._render_mesh = {}
        if cfg.reverse:
            img = img[:,::-1]
        return img

    import cv2

    if cfg.multistage:
        from diffsolver.program.scenes.visiontask import TaskSeq
        task = TaskSeq(int(cfg.taskname.split('task')[-1]))
        #print(task)
        for i in range(task.num_stages):
            if len(cfg.render_stages) > 0 and i not in cfg.render_stages:
                continue
            state = task.fetch_stage(i)[0]
            kwargs = {}
            if i < len(cfg.stage_tools):
                kwargs = cfg.stage_tools[i]
            img = render_state(state, **kwargs)
            cv2.imwrite(os.path.join(cfg.output, f'{i:03d}.png'), img[..., [2,1,0]])

        return

    trajs = torch.load(cfg.path + '/trajs.pt')

    p = os.listdir("examples/single_stage_dev")
    goal_state = None
    print('goal', cfg.goal_transparency)
    if cfg.render_goal:
        for i in p:
            if i.startswith(cfg.taskname+'_') or i.startswith(cfg.taskname+'.'):
                from diffsolver.config import DefaultConfig
                _c = OmegaConf.structured(DefaultConfig)
                f = OmegaConf.load("examples/single_stage_dev/"+i)
                from diffsolver.program.scenes import load_scene_with_envs
                _c = OmegaConf.merge(_c, f)
                scene_tuple = load_scene_with_envs(env, _c.scene)
                goal_state = env.get_state()
                img = render_state(goal_state, remove_tool=True)
                cv2.imwrite(os.path.join(cfg.output, f'goal.png'), img[..., [2,1,0]])


    idx = 0

    processor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    images = {}
    outs = []
    total = None
    total_mask = None

    state = None
    objects = []
    goal_img = None

    if goal_state is not None and cfg.goal_transparency < 1.:
        from llm.tiny import Scene
        goal_objs = Scene(env).get_object_list()
        objects = [
            {
                'pcd': i.pcd(),
                'colors': i.get_color()[:, :3],
                'object_id': 'goal{}'.format(idx),
            }  for idx, i in enumerate(goal_objs)
        ]

        goal_img = render_state(trajs['states'][0], remove_tool=not cfg.set_tool_pos, objects=objects, set_tool_pos=cfg.set_tool_pos)[..., [2,1,0]]
        # start = start * (1-cfg.goal_transparency) + start2 * cfg.goal_transparency
    start = render_state(trajs['states'][0], remove_tool=not cfg.set_tool_pos, objects=(),
                        set_tool_pos=cfg.set_tool_pos, qpos=cfg.tool_pos)[..., [2,1,0]]
    if goal_img is not None:
        start = start * cfg.goal_transparency + goal_img * (1-cfg.goal_transparency)

    cv2.imwrite(os.path.join(cfg.output, f'start.png'), start)

    if cfg.render_mp:
        if os.path.isfile(cfg.path + '/mp.th'):
            mp = torch.load(cfg.path + '/mp.th')
            mp_videos = []
            for i, q in tqdm.tqdm(enumerate(mp), total=len(mp), desc='rendering'):
                mp_videos.append(
                    render_state(trajs['states'][0], set_tool_pos=True, qpos=q)
                )
            animate(mp_videos, os.path.join(cfg.output, 'mp.mp4'))



    for state in tqdm.tqdm(trajs['states'], total=len(trajs['states']), desc='rendering'):
        # state.X[:, 1] -= 0.03
        if cfg.frames is None or idx in cfg.frames:
            img = render_state(state)

            images[idx] = img
            mask = processor.apply(img)

            cv2.imwrite(os.path.join(cfg.output, f'{idx:03d}_mask.png'), mask)

            mask = mask > 100.

            if total is None:
                total = np.copy(img)
                total_mask = np.zeros_like(mask)
            else:
                threshold = cfg.threshold if isinstance(cfg.threshold, float) else cfg.threshold[idx]
                total[mask] = (total * (1-threshold) + img * threshold)[mask]
                total_mask = total_mask | mask

            outs.append(img)

            cv2.imwrite(os.path.join(cfg.output, f'{idx:03d}.png'), img[..., [2,1,0]])

        idx += 1

    assert state is not None
    img = render_state(state, remove_tool=True)
    # env.set_state(state)
    # gui.reset(env)
    # img = gui.capture()


    cv2.imwrite(os.path.join(cfg.output, f'final.png'), img[..., [2,1,0]])

    assert total is not None
    cv2.imwrite(os.path.join(cfg.output, f'total.png'), total[..., [2,1,0]])
    animate(outs, os.path.join(cfg.output, 'video.mp4'))

    
if __name__ == '__main__':
    main()