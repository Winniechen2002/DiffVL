from tools.utils import logger
from tools.config import Configurable

class Exp(Configurable):
    def __init__(self, cfg=None, x=1., y=2.):
        super().__init__()
        logger.configure('tmp', ['wandb'], config=self._cfg)

    def start(self):
        for i in range(10):
            logger.logkvs({'x': i, 'y': i**2})
            logger.dumpkvs()

exp = Exp()
exp.start()