import pickle
from mpm.cuda_env import CudaEnv
import numpy as np

cu_env = CudaEnv(PRIMITIVES=[], SIMULATOR=dict(n_particles=3000, dt=1.))
r = cu_env.renderer

with open('render_checkpoint.pkl', 'rb') as f:
    sdf, color, bbox = pickle.load(f)
with open('p.pkl', 'rb') as f:
    p_x = pickle.load(f)

from diffrl.utils import lookat
print(cu_env.simulator.states[0].body_pos.download().reshape(-1, 3))
print(cu_env.simulator.states[0].body_rot.download().reshape(-1, 4))

camera = lookat([0.5, 0.1, 0.5], np.pi/4, 0., radius=10)
print(camera)
cu_env.renderer.initialize_camera(*camera)
"""
cu_env.simulator.states[0].x.upload(p_x.reshape(-1, 3))

cu_env.simulator.particle_color.upload(np.zeros(3000, dtype=np.int32) +(((127<<8)+127)<<8)+127)
"""

import matplotlib.pyplot as plt
# r.bake_particles(cu_env.simulator, cu_env.simulator.states[0])
image = cu_env.renderer.render(cu_env.simulator, cu_env.simulator.states[0])

volume = r.sdf_volume.download().reshape(168, 168, 168)

plt.imshow(image[..., :3])
plt.savefig('output.png')