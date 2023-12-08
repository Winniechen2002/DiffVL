import numpy as np
from typing import Dict, Optional
from tools import CN
from tools.utils import totensor, tonumpy, lookat
from gym.spaces import Box

from mpm.simulator import MPMSimulator
from mpm.renderer import Renderer
from mpm.torch_wrapper import DiffModel
from .goal_env import GoalEnv
from .world_state import WorldState

from .end_effector import \
    Gripper, DoublePushers, Knife, Pusher, \
    Tool


class MultiToolEnv(GoalEnv):
    # only define the dynamics here. modify the observation through wrapper please.
    def __init__(
        self,
        cfg:Optional[CN]=None,
        # cfg_path='../../mpm/configs/plb_cuda.yml',
        sim_cfg=MPMSimulator.get_default_config(
            yield_stress=50.,
            ground_friction=0.9,
            E=5000.,
            nu=0.2,
            grid_size=(1., 1., 1.),
            gravity=(0., -1., 0.),
            max_steps=1024,
            n_particles=20000
        ),
        render_cfg:Optional[CN]=None, #Renderer.get_default_config(),
        camera_cfg=dict(center=(0.5, 0.2, 0.5), theta=np.pi/4, phi=0., radius=3., primitive=1),

        MAX_N_BODIES = 10,
        gripper_cfg: Optional[CN]=None
    ):
        super().__init__()
        #n_bodies, kwargs = self.parse_tools(PRIMITIVES)
        self.MAX_N_BODIES = 10
        import os

        self.simulator = MPMSimulator(self.MAX_N_BODIES, cfg=sim_cfg)
        self.device = self.simulator.device
        if 'CUDA_VISIBLE_DEVICES' in os.environ and self.device is not None:
            self.device = 'cuda:'+str(os.environ['CUDA_VISIBLE_DEVICES'].split(',').index(self.device.split(':')[-1]))
        else:
            self.device = 'cuda:0'

        self.diffsim = DiffModel(self, return_grid=(), return_svd=False)
        self.renderer = None

        if self.renderer is None:
            self.setup_renderer()
        
        # tools
        self.tools: Dict[str, Tool] = {
            'Gripper': Gripper(self.simulator, cfg=gripper_cfg),
            'Pusher': Pusher(self.simulator, size=(0.2, 0.03, 0.2), lv=(0.012, 0.012, 0.012), friction=1., softness=666., K=0.0, mode='Box'),
            'DoublePushers': DoublePushers(self.simulator, size=(0.2, 0.03, 0.2), lv=(0.012, 0.012, 0.012), friction=1., softness=666., K=0.0, mode='Box'),
            'Knife': Knife(self.simulator, size=(0.01, 0.1, 0.2), lv=(0.012, 0.012, 0.012), friction=1., softness=666., K=0.0, mode='Box'),
            'Rolling_Pin': Knife(self.simulator, size=(0.03, 0.3), lv=(0.012, 0.012, 0.012), av=(0.0, 0.1, 0.), friction=1., softness=666., K=0.0, mode='Capsule'),
            # 'Fingers': DoublePushers(self.simulator, size=(0.2, 0.03, 0.2), lv=(0.012, 0.012, 0.012), friction=1., softness=666., K=0.0, mode='Box'),
        }
        for tool_name, tool in self.tools.items():
            tool.name = tool_name
        self.tool_cur: Optional[Tool] = None
        self.action_space: Box = None

        
        self.E_nu_yield = None

        
        self.initial_state = None
    
    def setup_renderer(self):
        self.renderer = Renderer(cfg=self._cfg.render_cfg)
        cam = self._cfg.camera_cfg

        self.renderer.setRT(*lookat(cam.center, cam.theta, cam.phi, cam.radius))
        self.renderer.particle_color = None #self._particle_color


    # I think the following function is not necessary.
    # def get_empty_state(self, n=None, tool_name='Gripper', **kwargs):
    #    return WorldState.get_empty_state(n, tool_name, **kwargs)
    def extend(self, T):
        self.simulator.extend(T)

    def set_state(self, state: WorldState, index=0, device='cuda:0', requires_grad=False, **kwargs):

        self.initial_state = state # preserve the initial state so that we can visit those info easily ..

        self.device = device
        self._requires_grad = requires_grad

        self.tool_cur = self.tools[state.tool_name]

        self.tool_cur.reset(state.tool_cfg)

        self.action_space = Box(-1, 1, self.tool_cur.get_action_shape())

        qpos = totensor(state.qpos, self.device)
        #print("st qpos,qpos",state.qpos,qpos)
        self.tool_cur.set_state(qpos, index)

        fk_result = tonumpy(self.tool_cur.forward_kinematics(qpos))
        # print(fk_result)
        fk_result = np.concatenate(fk_result, axis=-1)

        rigid_bodies = state.rigid_bodies
        if rigid_bodies is None:
            rigid_bodies = fk_result
        else:
            assert np.allclose(rigid_bodies, fk_result) 

        self.simulator.set_state(index, list([state.X, state.V, state.F, state.C]) + list(rigid_bodies))
        if state.ids is not None:
            self.simulator.set_object_id(state.ids)
        self.simulator.set_softness(state.softness)

        E_nu_yield = state.E_nu_yield
        if E_nu_yield is None:
            E_nu_yield = np.zeros((len(state.X), 3)) - 1

        E_nu_yield[E_nu_yield[:, 0] < -0.5, 0] = self.simulator._cfg.E
        E_nu_yield[E_nu_yield[:, 1] < -0.5, 1] = self.simulator._cfg.nu
        E_nu_yield[E_nu_yield[:, 2] < -0.5, 2] = self.simulator._cfg.yield_stress

        self.E_nu_yield = E_nu_yield
        from .soft_utils import mu_lam_by_E_nu
        properties = np.stack([*mu_lam_by_E_nu(E_nu_yield[:, 0], E_nu_yield[:, 1]), E_nu_yield[:, -1]], -1)
        self.simulator.set_particle_properties(properties)

        self.renderer.particle_color = state.color

        self._idx = 0
        if self._requires_grad:
            self.diffsim.zero_grad(**kwargs)
        self._obs = self.diffsim.get_obs(0, device)
        #return self.get_obs()

    
    def get_state(self, index=0) -> WorldState:
        assert isinstance(index, int)
        assert self.tool_cur is not None
        tmp = self.simulator.get_state(index)
        return WorldState(
            *tmp[:4], self.tool_cur.get_state(index), self.tool_cur.name,
            np.array(tmp[4:]), self.simulator.get_object_id(device='numpy'),
            self.renderer.particle_color.copy(),
            self.simulator.get_softness(), self.E_nu_yield, self.tool_cur._cfg, # config of the tool ..
        )

    def get_obs(self):
        obs =  self.diffsim.wrap_obs(self._obs)
        # print('asdasdasd',self.tool_cur.qpos_list)
        assert self.tool_cur is not None
        obs['qpos'] = self.tool_cur.qpos_list[self.get_cur_steps()]
        from diffsolver.program.types.obs import OBSDict
        obs: OBSDict = obs
        return obs

    def reset_state_goal(self, states, goals):
        self.goals = goals
        self.set_state(states)
        return self.get_obs()

    def get_cur_steps(self):
        return self._idx * self.simulator.substeps

    def _simulate(self, action, delta_velocity=None):
        start_substeps = self.get_cur_steps()
        nxt_step = start_substeps + self.simulator.substeps

        assert nxt_step < self.simulator.max_steps,\
            f"{nxt_step} exceed max steps {self.simulator.max_steps} in grad mode"

        self.tool_cur.qstep(start_substeps, action)

        pos, rot = self.tool_cur.forward_kinematics(self.tool_cur.qpos_list[start_substeps+1:nxt_step+1])
        self._obs = self.diffsim.diff_forward(self._idx, pos, rot, delta_velocity, *self._obs)
        self._idx += 1
        
        if not self._requires_grad:
            self._obs = [i.detach() for i in self._obs]
            self.simulator.copy_frame()
            self.tool_cur.copy_frame()
            self._idx = 0

    @property
    def batch_size(self):
        return 1

    def _render_rgb(self, batch_index=0, index=None, **kwargs):
        assert batch_index == 0

        index = self.get_cur_steps() if index is None else index
        self.simulator.set_color(self.renderer.particle_color)
        img = self.renderer.render(self.simulator, self.simulator.states[index], **kwargs)
        if img.shape[-1] >= 3:
            img = (img[:, :, :3].clip(0, 1) * 255).astype(np.uint8)
        return img
    
    def step(self, action, delta_velocity=None):
        action = totensor(action, self.device)
        if delta_velocity is not None:
            delta_velocity = totensor(delta_velocity, self.device)
        self._simulate(action, delta_velocity)
        return self.get_obs(), 0, False, {} 


    def render_state_rgb(self, state=None, pos=None, color=None):
        # only render state now
        def p():
            x = np.zeros(7)
            x[4] = 1
            return x

        n_bodies = [p() for _ in range(self.simulator.n_bodies)]
        if pos is not None:
            #state = self.get_state(-1)
            n = len(pos)
            state = WorldState.get_empty_state(n, self.tool_cur.name)
            state.X[:] = pos
            if color is not None:
                state.color[:] = color

        n = self.simulator.n_particles
        c = self.renderer.particle_color

        self.simulator.set_state(-1, list([state.X, state.V, state.F, state.C]) + list(n_bodies))

        self.simulator.n_particles = len(state.X)
        self.renderer.particle_color = state.color

        img = self._render_rgb(index=-1, primitive=False)
        self.simulator.n_particles = n
        self.renderer.particle_color = c
        return img