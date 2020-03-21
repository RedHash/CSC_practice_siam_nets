from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, CosineAnnealingLR
import config as cfg


class WarmupScheduler(_LRScheduler):

    def __init__(self, optimizer, warmup, last_epoch, after_warmup):
        self.optimizer = optimizer
        self.warmup = warmup
        self.after_warmup = after_warmup
        self.lrs0 = [p_group['initial_lr'] for p_group in self.optimizer.param_groups]
        self.timestep = last_epoch
        super().__init__(optimizer)

    def get_lr(self):
        return [(self.timestep / self.warmup) * lr0 for lr0 in self.lrs0]

    def step(self, epoch=None):
        if self.timestep < self.warmup:
            self.timestep += 1
            super().step(epoch)
        else:
            self.after_warmup.step(epoch)


def build_scheduler(optimizer, last_epoch):
    scheduler_creator = {
        'Exp': lambda opt: ExponentialLR(opt, gamma=cfg.SCHEDULER_GAMMA, last_epoch=last_epoch),
        'Cos': lambda opt: CosineAnnealingLR(opt, T_max=cfg.SCHEDULER_TMAX, last_epoch=last_epoch),
    }
    return WarmupScheduler(
        optimizer, warmup=cfg.SCHEDULER_WARMUP, last_epoch=last_epoch,
        after_warmup=scheduler_creator[cfg.SCHEDULER_AFTER_WARMUP](optimizer),
    )


if __name__ == '__main__':
    """ quick test """

    import torch.nn as nn
    from torch.optim import Adam
    from warnings import filterwarnings
    filterwarnings("ignore")

    net = nn.Sequential(
        nn.Linear(10, 10),
        nn.Linear(10, 10),
    )
    opt = Adam([
        {'params': [p for n, p in net.named_parameters() if n.startswith('0')], 'initial_lr': 0.1},
        {'params': [p for n, p in net.named_parameters() if n.startswith('1')], 'initial_lr': 0.01}
    ])
    scheduler = build_scheduler(opt)

    for i in range(10):
        print(f"epoch: {i + 1}")
        print(*[param['lr'] for param in opt.param_groups], sep=' | ', end='\n')
        scheduler.step()

    print("\nRESUMING...")
    scheduler = build_scheduler(opt, last_epoch=10)
    for i in range(10):
        print(f"epoch: {i + 1 + 10}")
        print(*[param['lr'] for param in opt.param_groups], sep=' | ', end='\n')
        scheduler.step()
