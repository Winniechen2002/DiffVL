from torch.multiprocessing import Process, Queue, Pipe, set_start_method

class AsyncPlbEnv(Process):
    def __init__(self, *args, cfg=None, **kwargs):
        set_start_method('spawn')
        Process.__init__(self)
        self.args = args
        self.cfg = cfg
        self.kwargs = kwargs

        self.pipe, self.worker_pipe = Pipe()
        self.daemon = True
        self.start()
        self.init()

    def init(self):
        self.pipe.send(['init', None])
        self.action_space, = self.pipe.recv()

    def run(self):
        from llm.envs import MultiToolEnv
        from llm.tiny import SoftBody, Scene
        from llm.envs.test_utils import init_scene

        env = MultiToolEnv(*self.args, cfg=self.cfg, **self.kwargs)
        init_scene(env, 0)

        while True:
            op, data = self.worker_pipe.recv()
            if op == 'init':
                self.worker_pipe.send([env.action_space])

            elif op == 'get_scene':
                scene = Scene(env)
                scene.env = None
                self.worker_pipe.send(scene)
            elif op == 'render':
                self.worker_pipe.send(env.render(mode=data))
            elif op == 'step':
                self.worker_pipe.send(env.step(data))

            elif op == 'get_obs':
                self.worker_pipe.send(env.get_obs())
            else:
                raise NotImplementedError

                
                
    def get_scene(self):
        self.pipe.send(['get_scene', None])
        scene = self.pipe.recv()
        scene.env = self
        return scene
    
    def render(self, mode):
        self.pipe.send(['render', mode])
        return self.pipe.recv()

    def step(self, action):
        self.pipe.send(['step', action])
        return self.pipe.recv()

    def get_obs(self):
        self.pipe.send(['get_obs', None])
        return self.pipe.recv()

    def close(self):
        self.pipe.send(['close', None])
        self.pipe.close()
        
    
    def __del__(self):
        self.close()

