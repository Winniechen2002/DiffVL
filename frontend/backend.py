#  ./utils/novnc_proxy --vnc localhost:5901 --listen 127.0.0.1:5900
import os
import multiprocessing as mp
from multiprocessing import Process
# from diffsolver.set_render_devices import RENDER_DEVICES 
RENDER_DEVICES = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = RENDER_DEVICES
print(RENDER_DEVICES)

try:
   mp.set_start_method('spawn', force=True)
   print("spawned")
except RuntimeError:
   pass

backend = None

class Backend(Process):
    # asynchronize backend
    def __init__(self) -> None:
        super().__init__()
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.daemon = True

    def run(self):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = RENDER_DEVICES

        from llm.tiny import Scene
        from envs import MultiToolEnv
        from envs.test_utils import init_scene
        from frontend.scene_renderer import GUI

        gui = GUI()
        gui.refrehsed=None
        env = MultiToolEnv(sim_cfg=dict(max_steps=100))
        init_scene(env, 0)
        scene = Scene(env)
        gui.load_scene(scene)


        import time
        target_fps = 60
        while not gui._viewer.closed:  # Press key q to quit
            p = None

            while not self.input_queue.empty():
                p = self.input_queue.get()

            if p is not None:
                gui._viewer.enter_mode('normal')
                op, data = p
                if op == 'exit':
                    break
                elif op == 'upload':
                    print('loading...', data[1])
                    gui.load_scene_config(data[0], data[1])
                else:
                    raise NotImplementedError

            if gui.refreshed:
                self.output_queue.put(['refreshed', gui.refreshed])
                gui.refreshed=False

            start_time = time.time()

            gui.step()
            if time.time() - start_time < 1./target_fps:
                time.sleep(max(1./target_fps - (time.time() - start_time), 0))

            gui._scene.update_render()  # Update the world to the renderer

            gui.update_scene() # update the shape ..

            gui._viewer.render()

        del gui
        del env

    def quit(self):
        self.input_queue.put(('exit', None))

    def upload(self, data, id):
        self.input_queue.put(('upload', (data, id)))


def get_gui(create=True):
    global backend
    if backend is not None:
        print(backend.is_alive())
    if backend is not None and not backend.is_alive():
        del backend
        backend = None
    if create:
        if backend is None:
            backend = Backend()
            backend.start()
    return backend