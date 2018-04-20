

def _constant(p):
    return 1

def _linear(p):
    return 1-p

_schedules = {
    'linear': _linear,
    'constant': _constant
}

class Scheduler(object):

    def __init__(self, v, nvalues, schedule):
        self.n = 0.
        self.v = v
        self.nvalues = nvalues
        self.schedule = _schedules[schedule]

    def value(self):
        current_value = self.v * self.schedule(self.n / self.nvalues)
        self.n += 1.
        return current_value

    def value_steps(self, steps):
        return self.v * self.schedule(steps / self.nvalues)