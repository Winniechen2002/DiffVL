genome_txt = """box2.yml
1
0 0 keep contact tool obj0
0 0 keep reach obj0 25 16
"""

from llm.genetic.genome import Genome
from llm.genetic.trainer import Trainer


import torch
genome = Genome.deserialize(genome_txt)
print(genome.serialize())

trainer = Trainer.parse(genome, env=None)
#print(trainer.get_initial_action().shape)
# trainer.animate(trainer.get_initial_action())
trainer.optimize(None, path='tmp', max_iter=200)