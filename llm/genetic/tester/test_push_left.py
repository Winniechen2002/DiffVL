import yaml
import cv2
from llm.genetic.loader import load_prog
from llm.genetic.trainer import Trainer
from llm.genetic.utils import print_table

f = load_prog('dataset/genome/push_right_constrained.py')[0]

cfg = yaml.load(f.doc, Loader=yaml.FullLoader)
trainer = Trainer(f, None, cfg=cfg)

# cv2.imwrite('xx.png', trainer.env.render('rgb_array')[..., ::-1])

tables = trainer.forward(trainer.get_initial_action())
print(print_table(tables, print_mode=True))