import torch
from torch import nn, optim, distributions

from catanbot.rl.algos.base import OnPolicyRLAlgorithm
from catanbot.rl.utils.rl_utils import compute_multiagent_reward_to_go, compute_returns, compute_multiagent_gae
from catanbot.rl.utils.logging import Logger

class MultiagentA2C(OnPolicyRLAlgorithm):
    """
    A2C with multiple agents. We're going to split the rollouts into individual agent rollouts and do policy gradient on each.
    Each agent gets its own network, but we assume that there is a shared V function for all 4.
    """
    default_hyperparameters = {
        'vf_batch_size':128,
        'vf_itrs':1000,
        'vf_lr':1e-3,
        'policy_lr':1e-4,
        'entropy_coeff':0.01
    }
    def __init__(
            self,
            env,
            policies,
            vf,
            collector,
            eval_collector,
            
            discount = 0.99,
            reward_scale = 1.0,

            epochs = 100,
            rollouts_per_epoch = 1000,
            eval_rollouts_per_epoch = 1000, 
            eval_every = 5,
            
            vf_itrs = default_hyperparameters['vf_itrs'],
            vf_batch_size = default_hyperparameters['vf_batch_size'],

            entropy_coeff = default_hyperparameters['entropy_coeff'],

            policy_lr = default_hyperparameters['policy_lr'],
            vf_lr = default_hyperparameters['vf_lr'],

            opt_class = optim.Adam,
            ):
#        print('Dont forget to put all the policy opts back when you finish debugging single-agent.')
        super(MultiagentA2C, self).__init__(env, discount, reward_scale, epochs, rollouts_per_epoch, eval_rollouts_per_epoch, eval_every)
        self.policies = policies
        self.vf = vf
        
        self.vf_itrs = vf_itrs
        self.entropy_coeff = entropy_coeff

        self.policy_opts = [opt_class(policy.network.parameters(), lr=policy_lr) for policy in self.policies]
#        self.policy_opts = [opt_class(self.policies[0].network.parameters(), lr=policy_lr)]
        self.vf_opt = opt_class(self.vf.parameters(), lr=vf_lr)
        self.policy_lr = policy_lr
        self.vf_lr = vf_lr
        self.vf_batch_size = vf_batch_size
        self.collector = collector
        self.eval_collector = eval_collector

    def update(self, batch):
        """
        Given a batch of trajectories, update the networks using A2C.
        Note: Should implement generalized advantage estimation (as a util function). Will just regress to RTG for now.
        """
#        import pdb;pdb.set_trace()
        batch = compute_multiagent_reward_to_go(batch, self.discount)

#        import pdb;pdb.set_trace()
        for i in range(len(self.policies)):
            self.policy_update(batch, i)
            break #TODO add the multiagent back later.

        for idx2 in range(0, len(batch['observation']), 8):
            print(idx2)
            if (batch['observation'][0, :100] - batch['observation'][idx2, :100]).abs().sum() != 0:
                break

        #Critic update: MSE to rtg(s)
        for vf_i in range(self.vf_itrs):
            batch_mask = torch.randint(0, batch['observation'].shape[0], (self.vf_batch_size, ))
            
            vf_obs = self.vf(batch['observation'][batch_mask])
            rtg = batch['reward_to_go'][batch_mask] + batch['reward'][batch_mask]
            
            vf_loss = (vf_obs - rtg).pow(2).mean()

            self.vf_opt.zero_grad()
            vf_loss.backward()
            self.vf_opt.step()

        a = self.policies[0].action_dist(0, batch['observation'][0]).probs.view(54, 3)
        b = self.policies[0].action_dist(0, batch['observation'][idx2]).probs.view(54, 3)
        #Update logs with algo-specific stuff
        self.logger.record_item('Loss', vf_loss, prefix='Critic')
        self.logger.record_item('Initial Policy Dist', torch.cat([torch.arange(54).unsqueeze(1), a, b], dim=1), prefix='Policy_1')

    def policy_update(self, batch, pidx):
        """
        Perform each policy update separately
        """
#        import pdb;pdb.set_trace()
        pidx_mask = (batch['pidx'] == pidx)
        obs = batch['observation'][pidx_mask]
        act = batch['action'][pidx_mask]
        nobs = batch['next_observation'][pidx_mask]
        rew = batch['reward'][pidx_mask, pidx]
        term = batch['terminal'][pidx_mask]

        policy_dist = self.policies[pidx].action_dist('string that does nothing', obs)
        log_probs = policy_dist.log_prob(act)

        v_obs = self.vf(obs)[:, pidx]
        v_nobs = self.vf(nobs)[:, pidx]
        advantage = self.discount * v_nobs * (1-term) + rew - v_obs
        advantage = advantage.detach()

#        advantage2 = compute_multiagent_gae(batch, gamma=self.discount, lam=0.97, vf = self.vf)
#        advantage = advantage2[pidx_mask, pidx]

        policy_objective = (log_probs * advantage).mean()
        entropy_objective = policy_dist.entropy().mean()
        policy_loss = -(policy_objective + self.entropy_coeff*entropy_objective)

        self.policy_opts[pidx].zero_grad()
        policy_loss.backward()
        self.policy_opts[pidx].step()

        self.logger.record_item('Loss', policy_loss, prefix='Policy_{}'.format(pidx+1))
        self.logger.record_tensor('Entropy', policy_dist.entropy(), prefix='Policy_{}'.format(pidx+1))
        self.logger.record_tensor('Advantage', advantage, prefix='Policy_{}'.format(pidx+1))
        self.logger.record_item('Policy Top 5', policy_dist.probs[0].topk(5).indices/3, prefix='Policy_{}'.format(pidx+1))
        self.logger.record_item('Policy Acts', batch['action'][pidx:96:8].argmax(dim=1)/3, prefix='Policy_{}'.format(pidx+1))
        self.logger.record_tensor('V preds', v_obs, prefix='Policy_{}'.format(pidx+1))

    @property
    def networks(self):
        out = {
            'vf': self.vf,
            }
        for i, policy in enumerate(self.policies):
            out['policy_{}'.format(i+1)] = policy.network
        return out

    @property
    def hyperparameters(self):
        return {
            'vf_itrs':self.vf_itrs,
            'entropy_coeff':self.entropy_coeff,
            'policy_lr': self.policy_lr,
            'vf_lr': self.vf_lr,
            'vf_batch_size':self.vf_batch_size
            }
