import torch
import torch.nn.functional as F


class JobShopStates:
    """
    Job Shop state for parallel executions.

    Args:
        device: Where to create tensors.
    """
    # Number of features in the internal state
    size = 11

    def __init__(self, device: str = 'cpu', eps: float = 1e-5):
        self.num_j = None       # Number of jobs
        self.num_m = None       # Number of machines
        self.machines = None    # Machine assigment of each operation
        self.costs = None       # Cost of each operation
        self._factor = None     # Max cost
        self._eps = eps
        self._q = torch.tensor([0.25, 0.5, 0.75], device=device)
        #
        self.dev = device       # Tensor device
        self._bs_idx = None     # Batch index for accessing info
        self.bs = None          # Batch size
        #
        self.j_ct = None        # Completion time of jobs in the partial sol
        self.j_idx = None       # Index of active operation in jobs
        self.j_st = None
        self.last_ms = 0        #
        #
        self.m_ct = None        # Completion time of machines in the partial sol

    def init_state(self, ins: dict, bs: int = 1):
        """
        Initialize the state of the job shop.

        Args:
            ins: JSP instance.
            bs: Batch size (number of parallel states).
        Return:
            - The parallel states.
            - The mask of active operations for each state.
        """
        self.num_j, self.num_m = ins['j'], ins['m']
        self.machines = ins['machines'].view(-1).to(self.dev)
        self._factor = ins['costs'].max()
        self.costs = ins['costs'].view(-1).to(self.dev) / self._factor
        self.bs = bs
        self._bs_idx = torch.arange(bs, device=self.dev)
        #
        self.j_st = torch.arange(0, self.num_j * self.num_m, self.num_m,
                                 device=self.dev)
        self.j_idx = torch.zeros((bs, self.num_j), dtype=torch.int32,
                                 device=self.dev)
        self.j_ct = torch.zeros((bs, self.num_j), dtype=torch.float32,
                                device=self.dev)
        #
        self.m_ct = torch.zeros((bs, self.num_m), dtype=torch.float32,
                                device=self.dev)

        # Create the initial state and mask
        states = torch.zeros((bs, self.num_j, self.size), dtype=torch.float32,
                             device=self.dev)

        return states, self.mask.to(torch.float32)

    @property
    def mask(self):
        """
        Boolean mask that points out the uncompleted jobs.

        Return:
            Tensor with shape (bs, num jobs).
        """
        return self.j_idx < self.num_m

    @property
    def ops(self):
        """
        The index of active/ready operations for each job.
        Note that for the completed job the active operation is the one with
        index 0.

        Return:
            Tensor with shape (bs, num jobs).
        """
        return self.j_st + (self.j_idx % self.num_m)

    @property
    def makespan(self):
        """
        Compute makespan of solutions.
        """
        return self.m_ct.max(-1)[0] * self._factor

    def __schedule__(self, jobs: torch.Tensor):
        """ Schedule the selected jobs and update completion times. """
        _idx = self._bs_idx           # Batch index
        _ops = self.ops[_idx, jobs]   # Active operations of selected jobs
        macs = self.machines[_ops]    # Machines of active operations

        # Update completion times
        ct = torch.maximum(self.m_ct[_idx, macs],
                           self.j_ct[_idx, jobs]) + self.costs[_ops]
        self.m_ct[_idx, macs] = ct
        self.j_ct[_idx, jobs] = ct

        # Activate the following operation on job, if any
        self.j_idx[_idx, jobs] += 1

    def update(self, jobs: torch.Tensor):
        """
        Update the internal state.

        Args:
            jobs: Index of the job scheduled at the last step.
                Shape (batch size).
        """
        # Schedule the selected operations
        self.__schedule__(jobs)

        _idx = self._bs_idx  # Batch index
        job_mac = self.machines[self.ops]  # Machines of active ops
        mac_ct = self.m_ct.gather(1, job_mac)  # Completion time of machines
        curr_ms = self.j_ct.max(-1, keepdim=True)[0] + self._eps
        #
        n_states = -torch.ones((self.bs, self.num_j, self.size),
                               device=self.dev)
        n_states[..., 0] = self.j_ct - mac_ct
        # Distance of each job from quantiles computed among all jobs
        q_j = torch.quantile(self.j_ct, self._q, -1).T
        n_states[..., 1:4] = self.j_ct.unsqueeze(-1) - q_j.unsqueeze(1)
        n_states[..., 4] = self.j_ct - self.j_ct.mean(-1, keepdim=True)
        n_states[..., 5] = self.j_ct / curr_ms
        # Distance of each job from quantiles computed among all machines
        q_m = torch.quantile(self.m_ct, self._q, -1).T
        n_states[..., 6:9] = mac_ct.unsqueeze(-1) - q_m.unsqueeze(1)
        n_states[..., 9] = mac_ct - self.m_ct.mean(-1, keepdim=True)
        n_states[..., 10] = mac_ct / curr_ms

        return n_states, self.mask.to(torch.float32), self.last_ms - curr_ms

    def __call__(self, jobs: torch.Tensor, states: torch.Tensor):
        """
        Update the internal state at inference.

        Args:
            jobs: Index of the job scheduled at the last step.
                Shape (batch size).
        """
        # Schedule the selected operations
        self.__schedule__(jobs)

        _idx = self._bs_idx  # Batch index
        job_mac = self.machines[self.ops]  # Machines of active ops
        mac_ct = self.m_ct.gather(1, job_mac)  # Completion time of machines
        curr_ms = self.j_ct.max(-1, keepdim=True)[0] + self._eps
        #
        states[..., 0] = self.j_ct - mac_ct
        # Distance of each job from quantiles computed among all jobs
        q_j = torch.quantile(self.j_ct, self._q, -1).T
        states[..., 1:4] = self.j_ct.unsqueeze(-1) - q_j.unsqueeze(1)
        states[..., 4] = self.j_ct - self.j_ct.mean(-1, keepdim=True)
        states[..., 5] = self.j_ct / curr_ms
        # Distance of each job from quantiles computed among all machines
        q_m = torch.quantile(self.m_ct, self._q, -1).T
        states[..., 6:9] = mac_ct.unsqueeze(-1) - q_m.unsqueeze(1)
        states[..., 9] = mac_ct - self.m_ct.mean(-1, keepdim=True)
        states[..., 10] = mac_ct / curr_ms

        return self.mask.to(torch.float32)


@torch.no_grad()
def sampling(ins: dict,
             encoder: torch.nn.Module,
             decoder: torch.nn.Module,
             bs: int = 32,
             device: str = 'cpu'):
    """
    Sampling at inference.

    Args:
        ins: The instance to solve.
        encoder: Pointer Encoder.
        decoder: Pointer Decoder
        bs: Batch size  (number of parallel solutions to create).
        device: Either cpu or cuda.
    """
    num_j, num_m = ins['j'], ins['m']
    num_ops = num_j * num_m
    machines = ins['machines'].view(-1)
    encoder.eval()
    decoder.eval()

    # Reserve space for the solution
    sols = -torch.ones((bs, num_m, num_j), dtype=torch.long, device=device)
    _idx = torch.arange(bs, device=device)
    m_idx = torch.zeros((bs, num_m), dtype=torch.long, device=device)
    #
    jsp = JobShopStates(device)
    state, mask = jsp.init_state(ins, bs)

    # Encoding step
    embed = encoder(ins['x'], edge_index=ins['edge_index'])

    # Decoding steps
    for i in range(num_ops):
        #
        ops = jsp.ops
        logits = decoder(embed[ops], state) + mask.log()
        scores = F.softmax(logits, -1)

        # Select the next (masked) operation to be scheduled
        jobs = scores.multinomial(1, replacement=False).squeeze(1)

        # Add the selected operations to the solution matrices
        s_ops = ops[_idx, jobs]
        m = machines[s_ops]
        s_idx = m_idx[_idx, m]
        sols[_idx, m, s_idx] = s_ops
        m_idx[_idx, m] += 1
        # Update the context of the solutions
        mask = jsp(jobs, state)

    return sols, jsp.makespan


@torch.no_grad()
def greedy(ins: dict,
           encoder: torch.nn.Module,
           decoder: torch.nn.Module,
           device: str = 'cpu'):
    """
    Sampling at inference.

    Args:
        ins: The instance to solve.
        encoder: Pointer Encoder.
        decoder: Pointer Decoder
        device: Either cpu or cuda.
    """
    num_j, num_m = ins['j'], ins['m']
    num_ops = num_j * num_m
    machines = ins['machines'].view(-1)
    encoder.eval()
    decoder.eval()

    # Reserve space for the solution
    sols = -torch.ones((num_m, num_j), dtype=torch.long, device=device)
    m_idx = torch.zeros(num_m, dtype=torch.long, device=device)
    #
    jsp = JobShopStates(device)
    state, mask = jsp.init_state(ins, 1)

    # Encoding step
    embed = encoder(ins['x'], edge_index=ins['edge_index'])

    # Decoding steps
    for i in range(num_ops):
        #
        ops = jsp.ops
        logits = decoder(embed[ops], state) + mask.log()
        scores = F.softmax(logits, -1)

        # Select the next (masked) operation to be scheduled
        jobs = scores.max(1)[1]

        # Add the selected operations to the solution matrices
        s_ops = ops[0, jobs]
        m = machines[s_ops]
        s_idx = m_idx[m]
        sols[m, s_idx] = s_ops
        m_idx[m] += 1
        # Update the context of the solutions
        mask = jsp(jobs, state)

    return sols, jsp.makespan


def sample_training(ins: dict,
                    encoder: torch.nn.Module,
                    decoder: torch.nn.Module,
                    bs: int = 32,
                    device: str = 'cpu'):
    """
    Sample multiple trajectories while training.

    Args:
        ins: The instance to solve.
        encoder: Pointer Encoder.
        decoder: Pointer Decoder
        bs: Batch size (number of parallel solutions to create).
        device: Either cpu or cuda.
    """
    num_j, num_m = ins['j'], ins['m']
    num_ops = num_j * num_m
    encoder.train()
    decoder.train()

    # Reserve space for the solution
    trajs = -torch.ones((bs, num_ops), dtype=torch.long, device=device)
    ptrs = -torch.ones((bs, num_ops, num_j), dtype=torch.float32,
                       device=device)
    _idx = torch.arange(0, bs, device=device)
    #
    jsp = JobShopStates(device)
    state, mask = jsp.init_state(ins, bs)

    # Encoding step
    embed = encoder(ins['x'].to(device),
                    edge_index=ins['edge_index'].to(device))

    # Decoding steps
    for i in range(num_ops):
        # Generate logits and mak the completed jobs
        ops = jsp.ops
        logits = decoder(embed[ops], state) + mask.log()
        scores = F.softmax(logits, -1)

        # Select the next (masked) operation to be scheduled
        jobs = scores.multinomial(1, replacement=False).squeeze(1)

        # Add the node and pointers to the solution
        trajs[_idx, i] = jobs
        ptrs[_idx, i] = logits
        #
        state, mask, _ = jsp.update(jobs)

    return trajs, ptrs, jsp.makespan
