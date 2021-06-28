"""The Noam lr_scheduler used in "Attention is All you Need"
Implementation Reference: https://nlp.seas.harvard.edu/2018/04/03/attention.html
"""

class NoamOpt(object):
    # Optim wrapper that implements rate.
    def __init__(self, hidden_size, factor, warmup, optimizer, writer=None):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.hidden_size = hidden_size
        self._rate = 0
        self.writer = writer
        self.param_groups = optimizer.param_groups
        self.state_dict = optimizer.state_dict

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        if self.writer is not None:
            self.writer.add_scalar('lr', rate, self._step)

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.hidden_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()