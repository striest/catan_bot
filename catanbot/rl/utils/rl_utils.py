import torch

"""
List of basic functions (like computing reward-to-go) necessary for some RL algos.
"""

def compute_multiagent_reward_to_go(batch, discount):
    """
    Computes the reward-to-go of a batch of trajectories
    Expects the same batch format as the single-agent version with the following exceptions (N=number of agents)
    Reward is now a [t x N] tensor
    There is an additional pidx tensor of shape [t x N] that indicates which agent took the action.
    """
    N = batch['pidx'].max().long().item() + 1
    reward_to_go = torch.zeros(batch['reward'].shape, device=batch['reward'].device)
    running_rtg = torch.zeros(N, device=batch['reward'].device)

    for i in range(reward_to_go.shape[0]-1, -1 ,-1):
        if batch['terminal'][i]:
            running_rtg *= 0
        else:
            discounts = torch.ones(N, device=batch['reward'].device)
            discounts[batch['pidx'][i]] = discount
            running_rtg = discounts * running_rtg.clone() + batch['reward'][i+1].clone()
        reward_to_go[i] = running_rtg

    batch['reward_to_go'] = reward_to_go
    return batch

def compute_reward_to_go(batch):
    """
    Computes reward-to-go of a batch of trajectories.
    Expects trajs to be in the following form: {'observation':torch.Tensor[t x obs_dim], 'action':torch.Tensor[t x act_dim], 'terminal':torch.Tensor[t], 'reward':torch.Tensor[t], 'next_observation':torch.Tensor[t x obs_dim]}
    This function will add the reward-to-go at each timestep as 'reward_to_go':torch.Tensor[t] to the batch dict.
    """
    #Note: There's probably a better way to do all this in parallel.
    #Until then, rtg[i] = 0 if t[i] else rtg[i+1] + r[i+1]

    running_rtg = 0
    reward_to_go = torch.zeros(batch['reward'].shape, device = batch['reward'].device)

    for i in range(reward_to_go.shape[0]-1, -1, -1):
        if batch['terminal'][i]:
            running_rtg = 0
        else:
            running_rtg += batch['discounted_reward'][i + 1]
        reward_to_go[i] = running_rtg

    batch['reward_to_go'] = reward_to_go
    return batch    

def compute_multiagent_returns(batch):
    """
    Just like single agent compute returns, but returns an N-tensor of returns.
    """
    N = batch['pidx'].max().long().item() + 1
    rets = []
    running_ret = torch.zeros(N, device=batch['reward'].device)

    for i in range(batch['reward'].shape[0]):
        running_ret += batch['reward'][i]
        
        if batch['terminal'][i]:
            rets.append(running_ret)
            running_ret = 0

    return rets

def compute_returns(batch):
    """
    Computes the returns of a batch of trajectories and returns it as a tensor
    """
    rets = []
    running_ret = 0

    for i in range(batch['reward'].shape[0]):
        running_ret += batch['reward'][i]
        
        if batch['terminal'][i]:
            rets.append(running_ret)
            running_ret = 0

    return rets
#    return torch.tensor(rets, device = batch['reward'].device)

def compute_multiagent_gae(batch, gamma, lam, vf):
    """
    Computes Generalizes Advantage Estimation (Schulman et al. 2015) in a multiagent setting.
    Like the other multiagent variants of the utils here, assumes:
    1. Reward is now [t x N]
    2. There is a pidx tensor of shape [t x N]
    """
    N = batch['pidx'].max().long().item() + 1
    advantages = torch.zeros(batch['reward'].shape, device = batch['reward'].device)
    v_obs = vf(batch['observation']).detach()
    v_nobs = vf(batch['next_observation']).detach()
    rew = batch['reward']
    tds = gamma * v_nobs + rew - v_obs

    running_adv = torch.zeros(N, device=batch['reward'].device)

    for i in range(advantages.shape[0]-1, -1, -1):
        if batch['terminal'][i]:
            running_adv *= 0.
        else:
            discounts = torch.ones(N, device=batch['reward'].device)
            discounts[batch['pidx'][i]] = gamma
            lambdas = torch.ones(N, device=batch['reward'].device)
            lambdas[batch['pidx'][i]] = lam
            running_adv = tds[i] + (discounts * lambdas) * running_adv
        advantages[i] = running_adv
    return advantages

def compute_gae(batch, gamma, lam, vf):
    """
    Does Generalized advantage estimation (Schulman et al. 2015) for the batch. (Can't do the nice cumsum trick from spinning up because there are multiple trajs in a batch)
    """
    advantages = torch.zeros(batch['reward'].shape, device = batch['reward'].device)
    v_obs = vf(batch['observation']).detach()
    v_nobs = vf(batch['next_observation']).detach()
    rew = batch['reward']
    tds = gamma * v_nobs + rew - v_obs
    running_adv = 0
    for i in range(advantages.shape[0]-1, -1, -1):
        if batch['terminal'][i]:
            running_adv = 0.
        else:
            running_adv = tds[i] + (gamma * lam) * running_adv
        advantages[i] = running_adv
    return advantages
    

def select_actions_from_q(q, acts):
    """
    Given a tensor of q values batched as [batch x action] and a list of action_idxs of [batch], select the corresponding action from the tensor
    """
    return q[torch.arange(q.shape[0]), acts].unsqueeze(1)

#From rlkit: https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/pytorch_util.py
def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
