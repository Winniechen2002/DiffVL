"""
data structure to store the returned value of the constraint function
"""
from typing import List, TypeVar
import torch

T = TypeVar('T', bound='Constraint')

class Constraint:
    def __init__(
        self, 
        loss: torch.Tensor, 
        sat: torch.Tensor,
        code=None,
        info=None,
        sync: str|None=None,
        weight=1.,
        is_constraint=False,
    ) -> None:
        assert isinstance(loss, torch.Tensor)
        assert sat.dtype == torch.bool

        self.loss = loss * weight
        self.sat = sat
        self.code = code
        self.info = info
        self.sync = sync
        self.is_constraint = is_constraint

    def __and__(self, other):
        return AndConstraint(self, other)

    def tolist(self):
        return [{
            'loss': self.loss,
            'sat': self.sat,
            'code': self.code,
            'info': self.info,
            'sync': self.sync,
        }]

    def all(self) -> List["Constraint"]:
        return [self]

    def synchronize(self):
        raise NotImplementedError

class AndConstraint(Constraint):
    def __init__(
        self, *cons_list: Constraint #A: Constraint, B: Constraint
    ) -> None:
        total_loss = sum([c.loss for c in cons_list], torch.tensor(0., dtype=torch.float32))
        total_sat = torch.tensor(True, dtype=torch.bool)
        for c in cons_list:
            total_sat = total_sat and c.sat
        super().__init__(total_loss, total_sat)
        self.elements: List[Constraint] = list(cons_list)

    def __str__(self) -> str:
        return '&'.join([str(c) for c in self.elements])

    def tolist(self):
        return sum([c.tolist() for c in self.elements], [])

    def synchronize(self):
        return " and ".join([c.synchronize() for c in self.elements])

    def all(self):
        return sum([c.all() for c in self.elements], [])


class LastConstraint(Constraint):
    def __str__(self) -> str:
        return f'last({self.code})'

    def synchronize(self):
        return f"so that {self.sync if self.sync is not None else self.code}"


class KeepConstraint(Constraint):
    def __str__(self) -> str:
        return f'keep({self.code})'

    def synchronize(self):
        return f"keep that {self.sync if self.sync is not None else self.code}"