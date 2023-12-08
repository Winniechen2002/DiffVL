import yaml
import cv2
from llm.genetic.loader import load_prog
from llm.genetic.trainer import Trainer
from llm.genetic.utils import print_table

f = load_prog('dataset/genome/taco.py')[0]
print(f(None))