import torch
from torch import nn, optim, distributions

from catanbot.rl.algos.base import OffPolicyRLAlgorithm
from catanbot.rl.utils.rl_utils import select_actions_from_q, soft_update_from_to

class DQN(OffPolicyRLAlgorithm):
    default_hyperparameters = {
                    'qf_itrs': 200,
                    'qf_batch_size': 128,
                    'target_update_period': 1,
                    'target_update_tau': 0.001,
                    'epsilon': 0.02,
                    'qf_lr': 1e-4
                    }

    def __init__(
            self,
            env,
            qf,
            target_qf,
            replay_buffer,
            collector,
            eval_collector,

            discount = 0.99,
            reward_scale = 1.0,

            epochs = int(1e6),
            rollouts_per_epoch = 10,
            initial_rollouts = 0,
            eval_rollouts_per_epoch = 10,
            eval_every = 1,

            qf_itrs = default_hyperparameters['qf_itrs'],
            qf_batch_size = default_hyperparameters['qf_batch_size'],

            target_update_period = default_hyperparameters['target_update_period'],
            target_update_tau = default_hyperparameters['target_update_tau'],

            epsilon = default_hyperparameters['epsilon'],

            qf_lr = default_hyperparameters['qf_lr'],
            
            opt_class = optim.Adam
            ):
        super(DQN, self).__init__(env, discount, reward_scale, epochs, rollouts_per_epoch, eval_rollouts_per_epoch, eval_every, initial_rollouts, replay_buffer, qf_itrs, qf_batch_size)
        self.qf = qf
        self.target_qf = target_qf

        self.epsilon = epsilon

        self.target_update_period = target_update_period
        self.target_update_tau = target_update_tau

        self.qf_lr = qf_lr
        self.qf_opt = opt_class(self.qf.parameters(), lr=self.qf_lr)

        self.gpu = self.qf.gpu

        self.collector = collector
        self.eval_collector = eval_collector

    def update(self, batch):
        """
        Use double-DQN to update networks.
        """
        #DDQN loss: Let a' = argmax_a'[Q(s, a')]. gamma*Q_target(s', a') + r - Q(s, a)
        pidxs = batch['pidx'].squeeze()
        bidxs = torch.arange(self.qf_batch_size) 

        q_curr = self.qf(batch['observation'])[bidxs, pidxs]
        q_next = self.qf(batch['next_observation'])[bidxs, pidxs]
        q_target_next = self.target_qf(batch['next_observation'])[bidxs, pidxs]

        q_next_act_max = q_next.argmax(dim=1) #edited - switch back to argmax of q?
        q_target_next_max = q_target_next[bidxs, q_next_act_max]

        terms = (1 - batch['terminal'].float()).squeeze()
        rews = batch['reward'][bidxs, pidxs]

        q_target = self.discount * q_target_next_max * terms + rews
        q_target = q_target.detach()

        q_pred = (q_curr * batch['action']).sum(dim=1) #Actions are one-hot into Q, so this is easy

        qf_loss = (q_target - q_pred).pow(2).mean()

        self.qf_opt.zero_grad()
        qf_loss.backward()
        self.qf_opt.step()

        #For agents that didn't act, the Q values should be the same. max{Q(s, a)} = r + gamma * max{Q(s', a')}
        #Easiest way to do this is forward-pass everything and then set losses on actual actions to 0.
#        import pdb;pdb.set_trace()
        q_curr = self.qf(batch['observation'])
        q_target_next = self.target_qf(batch['next_observation'])
        rews = batch['reward']
        terms = 1 - batch['terminal'].float()

        q_curr_pred = q_curr.max(dim=2)[0]
        q_next_pred = q_target_next.max(dim=2)[0]

        q_target = self.discount * q_next_pred * terms + rews
        q_target = q_target.detach()

        td_error = q_target - q_curr_pred
        td_error[bidxs, pidxs] = 0.

        qf_loss = td_error.pow(2).mean()

        self.qf_opt.zero_grad()
        qf_loss.backward()
        self.qf_opt.step()

        if self.current_epoch % self.target_update_period == 0:
            soft_update_from_to(self.qf, self.target_qf, self.target_update_tau)
        
#        import pdb;pdb.set_trace()
        for idx2 in range(0, self.replay_buffer.n, 8):
#            print(idx2)
            if (self.replay_buffer.buffer['observation'][0, :100] - self.replay_buffer.buffer['observation'][idx2, :100]).abs().sum() != 0:
                break

        init_obs1 = self.replay_buffer.buffer['observation'][0]
        init_obs2 = self.replay_buffer.buffer['observation'][1]
        init_obs3 = self.replay_buffer.buffer['observation'][2]
        init_obs4 = self.replay_buffer.buffer['observation'][3]
        init_q1 = self.qf(init_obs1)[0, 0]
        init_q2 = self.qf(init_obs2)[0, 1]
        init_q3 = self.qf(init_obs2)[0, 2]
        init_q4 = self.qf(init_obs2)[0, 3]
        out = torch.cat([init_q1.view(54, 3), init_q2.view(54, 3), init_q3.view(54, 3), init_q4.view(54, 3)], dim=1)
        
        """
        init_obs1 = self.replay_buffer.buffer['observation'][0]
        init_obs21 = self.replay_buffer.buffer['observation'][0]
        init_obs31 = self.replay_buffer.buffer['observation'][0]
        init_obs41 = self.replay_buffer.buffer['observation'][0]
        init_q1 = self.qf(init_obs1)[0, 0]
        init_q21 = self.qf(init_obs21)[0, 1]
        init_q31 = self.qf(init_obs31)[0, 2]
        init_q41 = self.qf(init_obs41)[0, 3]
        out = torch.cat([init_q1.unsqueeze(0), init_q21.unsqueeze(0), init_q31.unsqueeze(0), init_q41.unsqueeze(0)], dim=0)
        """

        #Update logs whenever you collect new samples
        self.logger.record_item('Loss', qf_loss, prefix='QF')
        self.logger.record_item('_P1-2 Q Intitial', out, prefix='Performance') 
        
    @property
    def hyperparameters(self):
        return {
                'qf_itrs': self.qf_itrs,
                'qf_batch_size': self.qf_batch_size,
                'target_update_period': self.target_update_period,
                'target_update_tau': self.target_update_tau,
                'epsilon': self.epsilon,
                'qf_lr': self.qf_lr
            }

    @property
    def networks(self):
        return {
                'qf':self.qf,
                'target_qf':self.target_qf
            }
