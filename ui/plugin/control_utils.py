import numpy as np
from .plugin_base import Plugin


def camera_point_to_plane_dir(plugin: Plugin, mx, my):
    window = plugin.viewer.window
    out = np.linalg.inv(window.get_camera_projection_matrix()) @ np.array([mx, my, 1, 0])
    dir = plugin.viewer.camera_space_to_world_space([out[0], out[1], -1])
    return dir


def plane_xy(plugin: Plugin, cur_xyz):
    window = plugin.viewer.window
    P = window.get_camera_projection_matrix()
    # world space to camera space
    cur_xyz -= window.get_camera_position()

    cur = P @ np.append(plugin.viewer.world_space_to_camera_space(cur_xyz), 1)
    cur[:3]/=-cur[2]  #-?
    cur[0] = -cur[0]

    mx = cur[0]
    my = cur[1] + 10.

    out = np.linalg.inv(P) @ np.array([mx, my, 1, 0])
    dir = plugin.viewer.camera_space_to_world_space([out[0], out[1], -1])


    y_dir = (dir - cur_xyz)[:2]
    assert np.linalg.norm(y_dir) > 1e-5, "lookdown view not implemented .."
    y_dir = -y_dir / np.linalg.norm(y_dir)

    # x_dir = x_dir / np.linalg.norm(x_dir)
    x_dir = np.cross(np.append(y_dir, 0), [0, 0, 1])[:2]
    return np.array([x_dir, y_dir])[:,:2]



def track_mouse(plugin: Plugin, cur_height_fn = lambda: 0, track_cond=lambda window: window.mouse_down(1)):
    window = plugin.viewer.window
    mx, my = np.array(window.mouse_position)
    # print(cur_height_fn)
    # print(mx, my)
    if plugin.viewer.is_mouse_available(mx, my):
        if track_cond(window):
            cur_height = cur_height_fn()
            ww, wh = window.size
            mx = (mx / ww - 0.5)*2
            my = (my / wh - 0.5)*2


            dir = camera_point_to_plane_dir(plugin, mx, my)

            pos = window.get_camera_position()
            coord =  (cur_height - pos[2]) / dir[2] * dir + pos # TODO: cur_height ..
            return coord[:2], dir, pos
    return None

# def track_mouse_ray(plugin: Plugin, cur_height_fn = lambda: 0, track_cond=lambda window: window.mouse_down(1)):
#     window = plugin.viewer.window
#     mx, my = np.array(window.mouse_position)
#     if plugin.viewer.is_mouse_available(mx, my):
#         if track_cond(window):
#             cur_height = cur_height_fn()
#             ww, wh = window.size
#             mx = (mx / ww - 0.5)*2
#             my = (my / wh - 0.5)*2


#             dir = camera_point_to_plane_dir(plugin, mx, my)
#             pos = window.get_camera_position()
#             coord = (cur_height - pos[2]) / dir[2] * dir + pos
#             return dir, pos
#     return None

def track_mouse_mover(plugin: Plugin, cur_height_fn = lambda: 0): #lambda window: window.mouse_down(1)):
    window = plugin.viewer.window
    mx, my = np.array(window.mouse_position)
    if plugin.viewer.is_mouse_available(mx, my):
        cur_height = cur_height_fn()
        #print(cur_height)
        ww, wh = window.size
        #print(ww,wh,mx,my)
        mx = (mx / ww - 0.5)*2
        my = (my / wh - 0.5)*2
        #print(plugin.viewer.window.get_camera_projection_matrix())
        #print(plugin.viewer.window.get_camera_rotation())
        dir = camera_point_to_plane_dir(plugin, mx, my)
        #print("dir",dir)
        pos = window.get_camera_position()
        coord = (cur_height - pos[2]) / dir[2] * dir + pos
        return coord[:2], dir, pos
    return None, None, None

def track_mouse_mover3d(plugin: Plugin, cur_height_fn = lambda: 0): #lambda window: window.mouse_down(1)):
    window = plugin.viewer.window
    mx, my = np.array(window.mouse_position)
    if plugin.viewer.is_mouse_available(mx, my):
        cur_height = cur_height_fn()
        #print(cur_height)
        ww, wh = window.size
        #print(ww,wh,mx,my)
        mx = (mx / ww - 0.5)*2
        my = (my / wh - 0.5)*2
        #print(plugin.viewer.window.get_camera_projection_matrix())
        #print(plugin.viewer.window.get_camera_rotation())
        dir = camera_point_to_plane_dir(plugin, mx, my)
        #horizontal = camera_point_to_plane_dir(plugin, 0, 0)
        #print("dir",dir)
        #horizontal = horizontal / np.sqrt((horizontal**2).sum().item())
        pos = window.get_camera_position()
        coord = (cur_height - pos[2]) / dir[2] * dir + pos
        return coord, dir, pos
    return None, None, None

def track_mouse_pos(plugin: Plugin, cur_height_fn = lambda: 0):
    window = plugin.viewer.window
    mx, my = np.array(window.mouse_position)
    if plugin.viewer.is_mouse_available(mx, my):
        cur_height = cur_height_fn()
        ww, wh = window.size
        mx = (mx / ww - 0.5)*2
        my = (my / wh - 0.5)*2
        dir = camera_point_to_plane_dir(plugin, mx, my)
        pos = window.get_camera_position()
        coord =  (cur_height - pos[2]) / dir[2] * dir + pos # TODO: cur_height ..
        return coord[:2]
    return None

def control1d(plugin: Plugin):
    window = plugin.viewer.window
    action = np.zeros(4)

    if window.key_press('i'):
        action[2] += 1

    return action

def control3d(plugin: Plugin):
    window = plugin.viewer.window
    action = np.zeros(4)

    if window.key_press('w'):
        action[1] += 1

    if window.key_press('s'):
        action[1] -= 1

    if window.key_press('d'):
        action[0] += 1

    if window.key_press('a'):
        action[0] -= 1

    if window.key_press('i'):
        action[2] += 1

    if window.key_press('k'):
        action[2] -= 1

    if window.key_press('j'):
        action[3] += 1

    if window.key_press('l'):
        action[3] -= 1

    return action


def control6d(plugin: Plugin):
    window = plugin.viewer.window
    action = np.zeros(6)

    if window.key_press('w'):
        action[2] += 1

    if window.key_press('s'):
        action[2] -= 1

    if window.key_press('d'):
        action[0] += 1

    if window.key_press('a'):
        action[0] -= 1

    # up, down

    if window.ctrl and window.key_press('j'):
        action[4] += 1
    elif window.key_press('j'):
        action[3] += 1

    if window.ctrl and window.key_press('l'):
        action[4] -= 1
    elif window.key_press('l'):
        action[3] -= 1

    if window.ctrl and window.key_press('i'):
        action[5] += 1
    elif window.key_press('i'):
        action[1] += 1

    if window.ctrl and window.key_press('k'):
        action[5] -= 1
    elif window.key_press('k'):
        action[1] -= 1

    return action

    

    
def normal_monitor(viewer):
    self = viewer
    speed_mod = viewer.speed_mod
    if self.window.mouse_down(1):
        x, y = self.window.mouse_delta
        if self.focused_entity:
            self.arc_camera_controller.rotate_yaw_pitch(
                -self.rotate_speed * speed_mod * x,
                self.rotate_speed * speed_mod * y,
            )
        else:
            self.fps_camera_controller.rotate(
                0,
                -self.rotate_speed * speed_mod * y,
                self.rotate_speed * speed_mod * x,
            )

    if self.window.mouse_down(2):
        x, y = self.window.mouse_delta
        self.focus_entity(None)
        self.fps_camera_controller.move(
            0,
            self.rotate_speed * speed_mod * x,
            self.rotate_speed * speed_mod * y,
        )

    wx, wy = self.window.mouse_wheel_delta
    if wx != 0:
        if self.focused_entity:
            self.arc_camera_controller.zoom(
                self.scroll_speed * speed_mod * wx
            )
        else:
            self.fps_camera_controller.move(
                self.scroll_speed * speed_mod * wx, 0, 0
            )

    if self.focused_entity:
        self.arc_camera_controller.set_center(self.get_focus_entity_position())
    elif self.focused_camera:
        cam_pose = self.get_camera_pose(self.focused_camera)
        rpy = quat2euler(cam_pose.q)
        self.set_camera_xyz(*cam_pose.p)
        self.set_camera_rpy(rpy[0], -rpy[1], -rpy[2])