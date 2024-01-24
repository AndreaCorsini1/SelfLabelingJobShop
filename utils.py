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
