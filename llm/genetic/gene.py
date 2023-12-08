# program stores the code structure
from llm.pl.program import Operator

class Gene:
    def __init__(
        self,
        op: Operator,
        mode: str = 'st',
        start: int = 0,
        duration: int = 0,
        stage_id=0,
        code=None
    ):
        assert mode in ['st', 'mean', 'keep']

        self.op = op
        self.mode = mode
        self.start = start
        self.duration = duration
        self.stage_id = stage_id
        assert duration >= 1
        if code is None:
            code = self.op.code
        self.code = code

    def add_start_stage(self, gap):
        return Gene(self.op, mode=self.mode, start=self.start + gap, duration=self.duration, stage_id=self.stage_id, code=self.code)

    def execute(self, scene):
        end = self.start + self.duration

        def f(t):
            scene.set_timestep(t+1)
            return self.op()

        tables = []

        end = min(scene.T(), end)
        if self.mode == 'st':
            ans = f(end-1).value
        elif self.mode == 'mean':
            all_out = []
            for j in range(self.start, end):
                all_out.append(f(j).value)
            ans = sum(all_out) / len(all_out)
        else:
            assert self.mode == 'keep'
            ans = None
            all_out = []
            # print(self.start, end)
            for j in range(self.start, end):
                p = f(j)
                if isinstance(p, bool):
                    ans = p
                    if not p:
                        break
                    continue
                all_out.append(p.value)
            if ans is None:
                import torch
                all_out = torch.stack(all_out)
                weights = torch.softmax(-all_out * 100, -1) # softmin, not worse than hardmin
                ans = (weights * all_out).sum()/(weights.sum() + 1e-9)

        assert ans is not None, "The constraint is not applicable. Please check the time limits."
        tables.append({
            'start': self.start, # switch to stage id ..
            'end': end,
            'mode': self.mode,
            'value': ans if not isinstance(ans, bool) else ans,
            'code': self.code,
            'stage': self.stage_id
        })

        return tables
