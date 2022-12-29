from torch.optim import Adam
from torchvision.models.resnet import resnet18
import torch
import matplotlib.pyplot as plt 
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def init_schedulers(init_lr, epochs, steps_per_epoch):
    model = resnet18(weights=None)

    optimizer = Adam(
        params=model.parameters(),
        lr=init_lr,
    )
    schedulers = {}
    # "decay the learning rate with the cosine decay schedule without restarts"
    # Register cosine annealing just to set the base_lr right
    # cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, epochs, eta_min=0, last_epoch=-1
    # )
    # schedulers['cosine_annealing'] = cosine_annealing

    # # Add warmup 
    # warmup = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer=optimizer,
    #     base_lr=0.,
    #     max_lr=init_lr,
    #     step_size_up=steps_per_epoch // 5,
    #     cycle_momentum=False
    # )

    # schedulers['warmup'] = warmup
    # # Add CyclicLR to escape local minima
    # cyclic_lr = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer=optimizer,
    #     base_lr=optimizer.param_groups[0]['lr'],
    #     max_lr=optimizer.param_groups[0]['lr'] / 10,
    #     step_size_up = steps_per_epoch // 2, 
    #     step_size_down = steps_per_epoch // 2,
    #     cycle_momentum=False,
    #     # last_epoch=steps_per_epoch // 5
    #     last_epoch=-1
    # )
    # # cyclic_annealing = torch.optim.lr_scheduler.ChainedScheduler([cyclicLR, cosine_annealing])
    # schedulers['cyclic_lr'] = cyclic_lr

    # clr = CosineAnnealingWarmupRestarts(
    #     optimizer=optimizer,
    #     first_cycle_steps = steps_per_epoch,
    #     cycle_mult = 1.,
    #     max_lr = init_lr,
    #     min_lr = init_lr / 10,
    #     warmup_steps = steps_per_epoch // 3,
    #     gamma = 1.,
    #     last_epoch = -1
    # )
    # schedulers['try'] = clr
    exponential = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.25)
    schedulers['exponential'] = exponential

    return optimizer, schedulers

def main():
    # epochs = 6
    # steps_per_epoch = 154 * 4
    # init_lr = 2.e-4
    # final_lr = 1.e-6
    epochs = 50
    steps_per_epoch = 10
    init_lr = 1e-6

    optimizer, schedulers = init_schedulers(init_lr, epochs, steps_per_epoch)

    steps = []
    lrs = []
    
    min_lr = 100
    max_lr = 0
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            # if step < steps_per_epoch // 5 and epoch == 0:
            #     schedulers['warmup'].step()
            # else:
            # schedulers['cyclic_lr'].step()
            # schedulers['try'].step()
            steps.append(step + (epoch + 1) * steps_per_epoch)
            lr = optimizer.param_groups[0]["lr"]
            lrs.append(lr)
            if lr > max_lr:
                max_lr = lr
            if lr < min_lr:
                min_lr = lr
            print(lr)
        
        schedulers['exponential'].step()
        # schedulers['linear'].step()
    print(f"min LR: {min_lr}")
    print(f"max LR: {max_lr}")

    print(sum(lrs) / len(lrs))
    plt.plot(steps, lrs)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(1,4))
    plt.savefig('deleteme.png')

if "__main__" in __name__:
    main()