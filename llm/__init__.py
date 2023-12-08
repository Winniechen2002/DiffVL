import os
LLMPATH = os.path.join(os.path.dirname(__file__))



if 'LLM_CACHE' in os.environ:
    CACHEPATH = os.environ['LLM_CACHE']
else:
    CACHEPATH = os.path.join(os.path.dirname(__file__), '.cache')
#from .envs import *
#from .lang import *
#from .ui import *