import yaml
import cv2
from llm.genetic.loader import load_prog
from llm.genetic.trainer import Trainer
from llm.genetic.utils import print_table

f = load_prog('../dataset/genome/lift.py')[0]
cfg = yaml.load(f.doc, Loader=yaml.FullLoader)
trainer = Trainer(f, None, cfg=cfg)

from llm.genetic.libtool import MotionPlanner
state = trainer.init_state
mp = MotionPlanner(trainer.env)

from tools.utils import totensor
from llm.tiny import Scene

scene = Scene(trainer.env)
obj = scene.obj(0)

#[0., 0., 0.]
#[0., 1., 0.]
#[0., 0., 1.]
#[1., 0., 0.]

mp.grasp(scene, obj, totensor([0., 0., 1.], device='cuda:0'), use_lvc=True)