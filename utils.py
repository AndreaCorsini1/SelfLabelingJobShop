import os
from datetime import datetime


class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class ObjMeter:
    def __init__(self):
        self.sum = {}
        self.count = {}

    def update(self, ins: dict, val: float):
        """
        Update with a new value for an instance.

        Args:
            ins: JSP instance.
            val: objective value (e.g. makespan) of the solution.
        Returns:
            None
        """
        shape = ins['shape']
        if shape not in self.sum:
            self.sum[shape] = val
            self.count[shape] = 1
        else:
            self.sum[ins['shape']] += val
            self.count[shape] += 1

    def __str__(self):
        out = ""
        for shape in sorted(self.sum):
            val = self.sum[shape] / self.count[shape]
            out += f"\t\t\t{shape:5}: AVG Obj={val:4.3f}\n"
        return out[:-1]

    @property
    def avg(self):
        """ Compute total average value regardless of shapes. """
        return sum(self.sum.values()) / sum(self.count.values()) if self.count \
            else 0


class Logger(object):

    def __init__(self, file_name: str = 'log'):
        #
        self.line = None
        if not os.path.exists('./output/logs'):
            os.makedirs('./output/logs')
        self.file_path = f"./output/logs/{file_name}_" +\
                         f"{datetime.now().strftime('%d-%m-%H:%M')}.txt"

    def train(self, step: int, loss: float, makespan: float):
        self.line = f"{step:4},{loss:.3f},{makespan:.3f}"

    def validation(self, objective: float, gap: float = 0.):
        self.line += f",{objective:.3f},{gap:.3f}"

    def flush(self):
        # Flush line
        with open(self.file_path, 'a+') as f:
            f.write(f"{datetime.now().strftime('%d-%m-%H:%M')},{self.line}\n")
