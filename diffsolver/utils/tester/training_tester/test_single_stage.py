import matplotlib.pyplot as plt
from omegaconf import DictConfig
from diffsolver.utils import MultiToolEnv, load_scene, build_wind_env
from diffsolver.engine import Engine
import gc

def test():
    env = MultiToolEnv()
    state = load_scene("wind_gripper.yml").state
    env.set_state(state)

    img = env.render('rgb_array')
    assert img is not None

    plt.imshow(img)
    plt.savefig("test.png")

    del env, state
    gc.collect()
    


def test_train(nstep=1):
    from omegaconf import OmegaConf
    from diffsolver.config import DefaultConfig

    default_conf: DictConfig = OmegaConf.structured(DefaultConfig)
    default_conf.merge_with(dict(trainer=dict(nsteps=nstep)))

    env = MultiToolEnv()
    engine = Engine(env, default_conf)
    engine.main()

    del engine, env
    gc.collect()

if __name__ == '__main__':
    test_train(nstep=100)