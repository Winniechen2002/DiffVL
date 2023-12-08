genome_txt = """box.yml
3
0 1 keep left tool obj0
0 0 keep Not contact tool obj0
1 -1 keep contact tool obj0
2 2 keep reach obj0 25 16
"""

from llm.genetic.genome import Genome
from llm.genetic.trainer import Trainer


import torch
genome = Genome.deserialize(genome_txt)
g = Genome.deserialize(torch.load('tmp/optim_state')['genome'])
genome = genome.update_actions(g.actions)


print(genome.serialize())

trainer = Trainer.parse(genome, env=None)
trainer.optimize(None, path='tmp', max_iter=200)