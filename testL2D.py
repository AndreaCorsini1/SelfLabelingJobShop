import torch
import numpy as np
import pandas as pd
import os
from L2D.Params import configs
from L2D.JSSP_Env import SJSSP
from L2D.actor_critic import ActorCritic
from L2D.utils import g_pool_cal
from time import time
from inout import load_dataset

#
N_JOBS_N = configs.Nn_j
N_MACHINES_N = configs.Nn_m
LOW = configs.low
HIGH = configs.high


def greedy(policy, ins: dict, device: str = 'cpu'):
    """
    Construct a solution for the input instance by following the policy.

    Args:
        policy: policy network.
        ins: JSP instance.
    Returns:
        The result as a dict.
    """
    # Initialize the environment with the instance
    env = SJSSP(n_j=ins['j'], n_m=ins['m'])
    adj, fea, candidate, mask = env.reset((ins['costs'].numpy(),
                                           ins['machines'].numpy()))
    ep_reward = -env.max_endTime

    # Prepare the policy
    policy.n_j = ins['j']
    policy.n_m = ins['m']
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size(
                                 [1, env.number_of_tasks, env.number_of_tasks]),
                             n_nodes=env.number_of_tasks,
                             device=device)

    # Construct the solution
    done = False
    t1 = time()
    while not done:
        #
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)

        with torch.no_grad():
            pi, _ = policy(x=fea_tensor,
                           graph_pool=g_pool_step,
                           padded_nei=None,
                           adj=adj_tensor,
                           candidate=candidate_tensor.unsqueeze(0),
                           mask=mask_tensor.unsqueeze(0))

            # Greedy selection
            _, index = pi.squeeze().max(0)
            action = candidate[index]

        # Update env
        adj, fea, reward, done, candidate, mask = env.step(action)
        ep_reward += reward

    ms = env.posRewards - ep_reward
    print(f'{ins["name"]} = {ms}')
    return {
        'NAME': ins['name'],
        'MS': ms,
        'GAP': (ms / ins['makespan'] - 1) * 100,
        'TIME': time() - t1
    }


def sampling(policy, ins: dict, beta: int = 128, device: str = 'cpu'):
    """
    Construct multiple solutions for the input instance by sampling from
    the policy.

    Args:
        policy: policy network.
        ins: JSP instance.
    Returns:
        The result as a dict.
    """
    # Initialize the environment with the instance
    env = SJSSP(n_j=ins['j'], n_m=ins['m'])
    policy.n_j = ins['j']
    policy.n_m = ins['m']
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size(
                                 [1, env.number_of_tasks, env.number_of_tasks]),
                             n_nodes=env.number_of_tasks,
                             device=device)

    # Construct the solutions
    best_ms = float('inf')
    t1 = time()
    for _ in range(beta):
        # Init environment
        adj, fea, candidate, mask = env.reset((ins['costs'].numpy(),
                                               ins['machines'].numpy()))
        ep_reward = -env.max_endTime
        done = False
        while not done:
            #
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)

            with torch.no_grad():
                pi, _ = policy(x=fea_tensor,
                               graph_pool=g_pool_step,
                               padded_nei=None,
                               adj=adj_tensor,
                               candidate=candidate_tensor.unsqueeze(0),
                               mask=mask_tensor.unsqueeze(0))

                # Random selection
                index = pi.squeeze().multinomial(1, replacement=False)
                action = candidate[index]

                # Update env
                adj, fea, reward, done, candidate, mask = env.step(action)
                ep_reward += reward

        # Save best makespan
        ms = env.posRewards - ep_reward
        if ms < best_ms:
            best_ms = ms
    #
    print(f'{ins["name"]} = {best_ms}')
    return {
        'NAME': ins['name'],
        'MS': best_ms,
        'GAP': (best_ms / ins['makespan'] - 1) * 100,
        'TIME': time() - t1
    }


if __name__ == '__main__':
    # Seed torch
    torch.manual_seed(12345)
    print(f'Running on {configs.device}: beta = {configs.beta}')

    # Load the model
    _policy = ActorCritic(n_j=0, n_m=0,
                          num_layers=configs.num_layers,
                          learn_eps=False,
                          neighbor_pooling_type=configs.neighbor_pooling_type,
                          input_dim=configs.input_dim,
                          hidden_dim=configs.hidden_dim,
                          num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                          num_mlp_layers_actor=configs.num_mlp_layers_actor,
                          hidden_dim_actor=configs.hidden_dim_actor,
                          num_mlp_layers_critic=configs.num_mlp_layers_critic,
                          hidden_dim_critic=configs.hidden_dim_critic,
                          device=configs.device)
    path = f'./L2D/SavedNetwork/{N_JOBS_N}_{N_MACHINES_N}_{LOW}_{HIGH}.pth'
    _policy.load_state_dict(torch.load(path, map_location=configs.device))

    # Load the benchmark
    result = []
    dataset = load_dataset(f'./benchmarks/{configs.benchmark}', device='cpu')
    for i, ins in enumerate(dataset):
        #
        if configs.beta == 1:
            result.append(greedy(_policy, ins, device=configs.device))
        else:
            result.append(sampling(_policy, ins, configs.beta, configs.device))

    # Save results
    print(result)
    if not os.path.exists('./output/L2D'):
        os.makedirs('./output/L2D')
    out_file = f'./output/L2D/{configs.benchmark}_' \
               f'{N_JOBS_N}x{N_MACHINES_N}_B{configs.beta}.csv'
    pd.DataFrame(result).to_csv(out_file, index=False, sep=',')
