# a fake 
import torch
from .trainer import Trainer


class FrankWolfe(Trainer):
    # this is not a frankwolfe algorithm ..
    # do project: https://vene.ro/blog/mirror-descent.html
    def __init__(self, genome, env, cfg=None, lmbda_lr=1., reg_proj=0.01, weight_penalty=0.001, constraint_threshold=0., clip_lmbda=None):
        super().__init__(genome, env)


    def init_optim(self, actions, cfg):
        self.actions_params = actions
        self.reward_optim = torch.optim.Adam([actions], lr=cfg.lr)
        self.constraint_optim = torch.optim.Adam([actions], lr=cfg.lr)

        self.mu = 1. # for augmented lagrangian ..
        self.lmbda = {}
        self.last_good = None


    def print_table_func(self, x, decimal=5):
        v = float(x['value'])
        if decimal is not None:
            v = ('{:.' + str(decimal) + 'f}').format(v)
        if 'lmbda' in x:
            v = v +  f"| {x['lmbda']:.3f}"
        return v


    def optimize(self, actions, **kwargs):
        output = self.forward(actions[:], requires_grad=True, **kwargs)

        loss = 0.
        constraint = 0.
        penalty = 0.

        unscaled_constraint = 0.

        for i in output:
            value = -i['value'] 
            if i['is_constrained']:
                key = str((i['code'], i['start'], i['end']))

                if key not in self.lmbda:
                    self.lmbda[key] = 1. # initialize the constraints
                if value > 0.:
                    constraint = constraint + value * self.lmbda[key] + value * value * self.mu/2

                    unscaled_constraint += value

                elif value < 0.:
                    penalty = penalty - torch.log(-value) * self._cfg.weight_penalty

                i['lmbda'] = self.lmbda[key]
                self.lmbda[key] = max(self.lmbda[key] + self._cfg.lmbda_lr * float(value), 0.5)
                if self._cfg.clip_lmbda is not None:
                    self.lmbda[key] = min(self.lmbda[key], self._cfg.clip_lmbda)
            else:
                loss = loss + torch.relu(value)

        reg_action = 0.
        if constraint <= self._cfg.constraint_threshold:
            self.reward_optim.zero_grad()
            if isinstance(loss, torch.Tensor) or isinstance(penalty, torch.Tensor):
                (loss + penalty).backward()
            self.reward_optim.step()
        else:
            # push the trajectory to satisfy the constraints ..
            self.constraint_optim.zero_grad()
            if self._cfg.reg_proj > 0. and self.last_good is not None:
                reg_action = torch.linalg.norm(actions - self.last_good)**2 * self._cfg.reg_proj
                constraint += reg_action  # do not be far away from the last good ..
            constraint.backward()
            self.constraint_optim.step()


        with torch.no_grad():
            actions.data[:] = torch.clamp(actions, -1, 1)
            if constraint <= 0.:
                self.last_good = actions.data[:]


        return {
            'info': {
                'penalty': float(penalty),
                'constraint': float(constraint),
                'reg_action': float(reg_action),
                'unscaled_constraint': float(unscaled_constraint),
            },
            'loss': float(loss),
            'tables': output,
            'creteria': float(unscaled_constraint) * 1000 + float(loss)
        }