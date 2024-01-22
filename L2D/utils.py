import torch
import numpy as np


def aggr_obs(obs_mb, n_node):
    # obs_mb is [m, n_nodes_each_state, fea_dim], m is number of nodes in batch
    idxs = obs_mb.coalesce().indices()
    vals = obs_mb.coalesce().values()
    new_idx_row = idxs[1] + idxs[0] * n_node
    new_idx_col = idxs[2] + idxs[0] * n_node
    idx_mb = torch.stack((new_idx_row, new_idx_col))
    # print(idx_mb)
    # print(obs_mb.shape[0])
    adj_batch = torch.sparse.FloatTensor(indices=idx_mb,
                                         values=vals,
                                         size=torch.Size([obs_mb.shape[0] * n_node,
                                                          obs_mb.shape[0] * n_node]),
                                         ).to(obs_mb.device)
    return adj_batch


def g_pool_cal(graph_pool_type, batch_size, n_nodes, device):
    # batch_size is the shape of batch
    # for graph pool sparse matrix
    if graph_pool_type == 'average':
        elem = torch.full(size=(batch_size[0]*n_nodes, 1),
                          fill_value=1 / n_nodes,
                          dtype=torch.float32,
                          device=device).view(-1)
    else:
        elem = torch.full(size=(batch_size[0] * n_nodes, 1),
                          fill_value=1,
                          dtype=torch.float32,
                          device=device).view(-1)
    idx_0 = torch.arange(start=0, end=batch_size[0],
                         device=device,
                         dtype=torch.long)
    # print(idx_0)
    idx_0 = idx_0.repeat(n_nodes, 1).t().reshape((batch_size[0]*n_nodes, 1)).squeeze()

    idx_1 = torch.arange(start=0, end=n_nodes*batch_size[0],
                         device=device,
                         dtype=torch.long)
    idx = torch.stack((idx_0, idx_1))
    graph_pool = torch.sparse.FloatTensor(idx, elem,
                                          torch.Size([batch_size[0],
                                                      n_nodes*batch_size[0]])
                                          ).to(device)

    return graph_pool


def data_generator(n_j, n_m, low, high):
    """
    Uniform instance generator.

    Args:
        n_j: number of jobs
        n_m: number of machines
        low: minimum processing time
        high: maximum processing time.
    Returns:
        - 2D matrix of processing times
        - 2D matrix of machines
    """
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0)

    # Permute rows (machines)
    ix_i = np.tile(np.arange(machines.shape[0]), (machines.shape[1], 1)).T
    ix_j = np.random.sample(machines.shape).argsort(axis=1)
    machines = machines[ix_i, ix_j]

    return times, machines


def override(fn):
    """
    override decorator
    """
    return fn


if __name__ == '__main__':
    # Generate data offline
    j = 20
    m = 10
    l = 1
    h = 99
    batch_size = 100
    seed = 200
    np.random.seed(seed)

    data = np.array([data_generator(n_j=j, n_m=m, low=l, high=h)
                     for _ in range(batch_size)])
    print(data.shape)
    np.save(f'./DataGen/generatedData{j}_{m}_Seed{seed}.npy', data)
