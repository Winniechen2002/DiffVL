import abc
from tools.config import Configurable, as_builder


@as_builder
class Scheduler(Configurable, abc.ABC):
    def __init__(self, cfg=None, init_value=1.) -> None:
        super().__init__()

        self.epoch = 0
        self.init_value = init_value
        self.value = init_value

    def step(self, epoch=None):
        if epoch is None:
            self.epoch += 1
            delta = 1
        else:
            assert epoch > self.epoch
            delta = epoch - self.epoch
            self.epoch = epoch

        return self._step(self.epoch, delta)


    @abc.abstractmethod
    def _step(self, cur_epoch, delta):
        pass


class ConstantScheduler(Scheduler):
    def __init__(self, cfg=None) -> None:
        super().__init__(cfg)

    def _step(self, cur_epoch, delta):
        return self.value


class ExponentialScheduler(Scheduler):
    def __init__(self, cfg=None, decay=0.99, min_value=0.) -> None:
        super().__init__(cfg)
        self.decay = cfg.decay
        self.min_value = cfg.min_value

    def _step(self, cur_epoch, delta):
        self.value = max(self.min_value, self.value * (self.decay ** delta))
        return self.value


class MultiStepScheduler(Scheduler):
    def __init__(self, cfg=None, milestones=None, gamma=0.1) -> None:
        super().__init__(cfg)
        self.milestones = milestones
        self.gamma = gamma

    def _step(self, cur_epoch, delta):
        value = self.init_value
        for i in self.milestones:
            if cur_epoch >= i:
                value = value * self.gamma
        return value