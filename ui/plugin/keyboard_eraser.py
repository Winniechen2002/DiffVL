from .plugin_base import Plugin
from llm.tiny import Scene
import numpy as np
import torch

class KeyboardEraser(Plugin):
    mode_name = 'keyboard_erase'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update = None
        self.next_shape = None
        # self.leave_mode()

    def enter_mode(self):
        self._old_paused = self.viewer.paused
        self._next_shape = None
        self.viewer.toggle_pause(True)
        self.viewer.toggle_eraser(True)

        # print('self.viewer.erase: ', self.viewer.eraser_scale, self.viewer.eraser_pos, self.viewer.eraser_dir)
        # viewer = self.viewer
        # viewer.window.cursor = False
        #viewer.set_camera_xyz(0.5, -0.8, 0.8)
        #viewer.set_camera_rpy(0, -1., 3.14 + 3.14/2)
    
    def leave_mode(self):
        self.wx = None
        self.target = None
        self.up = None
        self.left = None
        self.front = None

        self.viewer.toggle_eraser(False)
        self.viewer.eraser_scale = [1, 1, 1]
        self.viewer.eraser_pos = [1, 0, 0] 
        self.viewer.eraser_dir = [0.7071068, 0, 0, 0.7071068]
        self.viewer.update_eraser()
        
        if self.viewer.paused:
            self.viewer.toggle_pause(self._old_paused)
        # if self.update is not None:
        #     self.changes = {}
        #     for i, k in self.engine._soft_bodies.items():
        #         self.changes[i] = k['tensor'][:, :3].detach().cpu().numpy()

    def rendering(self):
        # listen to keys or trigger the mode
        # if self.engine.tool_name == 'Eraser':
        viewer = self.viewer
        #self.selected = self.engine.selected_tool()
        if viewer.window.key_press('h'): # and self.selected:
            # print('erasing')
            self._trigger()

    def erase(self):
        state = self.engine._env.get_state()
        # stt = scn.env.initial_state
        env = self.engine._env
        env.set_state(state)
        scn = Scene(env)
        # print(stt.X.shape)
        
        cen = [self.viewer.eraser_pos[0], self.viewer.eraser_pos[2], self.viewer.eraser_pos[1]]
        rad = 0.1 * self.viewer.eraser_scale[0]
        scn.check_soft_bodies(cen, rad)
        # inn = ((stt.X - cen) ** 2).sum(axis=1) 
        # # print(inn.shape, inn, self.viewer.eraser_scale[0], self.viewer.eraser_pos, 0.01 * self.viewer.eraser_scale[0] * self.viewer.eraser_scale[0])
        # inn = inn > 0.01 * self.viewer.eraser_scale[0] * self.viewer.eraser_scale[0]
        # # inn_ids = [_ for _ in range(inn.size) if not inn[_]]
        # #print(inn_ids)
        # stt.X = stt.X[inn]
        # stt.V = stt.V[inn]
        # stt.F = stt.F[inn]
        # stt.C = stt.C[inn]
        # stt.ids = stt.ids[inn]
        # stt.color = stt.color[inn]
        # stt.E_nu_yield = stt.E_nu_yield[inn]

        # scn.initial_state = scn.env.initial_state = stt
        # scn.env.set_state(stt)       

        # wow = {}
        # for k, tp in self.engine._soft_bodies.items():
        #     ind = [_ for _ in range(tp['N']) if tp['indices'][_] not in inn_ids]
        #     print('ind', ind)
        #     tp['tensor'] = tp['tensor'][ind]
        #     tp['indices'] = tp['indices'][ind]
        #     tp['N'] = tp['N'] - len(ind)
        #     tp['pcd'] = self.engine._env.get_obs()['pos'][tp['indices']]
        #     wow[tp['id']] = tp
        # for k, v in self.engine._soft_bodies.items():
        #     self.engine._scene.remove_particle_entity(v['pcd'])
        # self.engine._soft_bodies = wow
        # object_list = scn.get_object_list()
        # for j in object_list:
        #     tp = self.engine._add_sapien_particles(j)
        #     if tp['id'] == self.cur_shape['id']:
        #         self.cur_shape = tp
        #         tp['tensor'][N:, :3] = quaternion_apply(
        #             totensor(q, 'cuda:0'), self.shape['tensor'][:, :3] - com 
        #         ) + com + totensor(p, 'cuda:0')
        #         self.shape['tensor'][:,:3] = tp['tensor'][N:, :3]
        #         self.shape['indeces'] = tp['indices'][N:]

        self.engine.reload_scene(scn)

        # object_list = scn.get_object_list()
        # for i in object_list:
        #     self.engine._add_sapien_particles(i)

    def monitor(self):
        # what would happen when endering the mode ..
        import numpy as np
        from .control_utils import track_mouse
        window = self.viewer.window

        self.target = track_mouse(self, lambda: float(self.engine.obs['qpos'][1]))
        wx, wy = window.mouse_wheel_delta
        if wx != 0:
            self.wx = wx
        else:
            self.wx = None

        self.left = 0
        if self.viewer.window.key_press('a'):
            self.left = -0.03
        if self.viewer.window.key_press('d'):
            self.left = 0.03
        
        self.front = 0
        if self.viewer.window.key_press('s'):
            self.front = -0.03
        if self.viewer.window.key_press('w'):
            self.front = 0.03
        
        self.up = 0
        if self.viewer.window.key_press('i'):
            self.up = 0.03
        if self.viewer.window.key_press('k'):
            self.up = -0.03

        self.scale = 1
        if self.viewer.window.key_press('o'):
            self.scale = 1.2
        if self.viewer.window.key_press('l'):
            self.scale = 0.8

        self.rot = 0
        if self.viewer.window.key_press('u'):
            self.rot = -0.2
        if self.viewer.window.key_press('j'):
            self.rot = 0.2
        
        self.viewer.eraser_pos[0] += self.left
        self.viewer.eraser_pos[1] += self.front
        self.viewer.eraser_pos[2] += self.up
        self.viewer.eraser_scale[0] *= self.scale
        self.viewer.eraser_scale[1] *= self.scale
        self.viewer.eraser_scale[2] *= self.scale
        print('self.viewer.erase: ', self.viewer.eraser_scale, self.viewer.eraser_pos, self.viewer.eraser_dir)
        # self.eraser_scale[0] += self.left
        self.viewer.update_eraser()

        self.erase()

        if window.mouse_click(0):
            self.viewer.enter_mode("normal")

    def step(self, cur_action=None):
        # modify simulation ..

        # if self.target is not None:
        #     qpos = self.engine.obs['qpos'].detach().cpu().numpy()
        #     cur_action[[0, 2]] = (self.target - qpos[[0, 2]]).clip(-0.5, 0.5)

        # if self.wx is not None:
        #     cur_action[4] = -max(min(self.wx / 5, 0.5), -0.5)
        #     #cur_action = cur_action

        # if self.left is not None:
        #     cur_action[0] = self.left * 0.3   
            
        # if self.up is not None:
        #     cur_action[1] = self.up * 0.3   
            
        # if self.front is not None:
        #     cur_action[2] = self.front * 0.3
        
        return cur_action, None