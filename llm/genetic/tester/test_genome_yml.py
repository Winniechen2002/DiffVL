import os
from llm.genetic.utils import GENOMEPATH
from tools.config import CN
from llm.genetic.genome import Genome
from llm.genetic.trainer import Trainer


import torch

import yaml
with open(os.path.join(GENOMEPATH, 'test.yml'), 'r') as f:
    cfg = yaml.load(f, Loader=yaml.Loader)
print(cfg)


if 'optim_cfg' in cfg:
    optim_cfg = cfg['optim_cfg']
genome = Genome.deserialize(cfg)
print(genome.serialize())


#print(genome.serialize())
print(genome)

trainer = Trainer.parse(genome, env=None, **optim_cfg)
trainer.main_loop(None, path='tmp', max_iter=200)