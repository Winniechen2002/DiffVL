import os

RENDER_DEVICES = os.environ.get('RENDER_DEVICES', None)
if RENDER_DEVICES is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = RENDER_DEVICES