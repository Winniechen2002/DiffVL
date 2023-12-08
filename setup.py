from setuptools import setup

install_requires = [
    'scipy',
    'numpy',
    'torch',
    'opencv-python',
    'tqdm',
    # 'taichi',
    'gym',
    'tensorboard',
    'yacs>=0.1.8',
    'matplotlib',
    # 'descartes',
    #'shapely',
    #'natsort',
    # 'torchvision',
    'einops',
    #'alphashape',
    'transforms3d',
    'h5py',
    #'bezier',
    # 'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git',
    'chamferdist',
    'geomloss',
    #'open3d',
    'pydprint',
    # 'pyro-ppl',
    'moviepy>=1.0.3'
    'gitpython',
    'ninja',
    'wandb',
    #'pyvista', # maybe not needed
    #'pythreejs', # maybe not needed
    'torchtyping',
]


setup(
    name='concept',
    version='0.0.1',
    packages=['mpm'],
    install_requires=install_requires,
    py_modules=['tr', 'rl', 'mpm', 'solver', 'diff_skill', 'frontend']
)

# for RTX 30 series card, install pytorch with cudatoolkit 11 to support sm_86
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
