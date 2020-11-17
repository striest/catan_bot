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

            discount = 0.99,
            reward_scale = 1.0,

            epochs = int(1e6),
            rollouts_per_epoch = 10,
            initial_rollouts = 0,

            qf_itrs = default_hyperparameters['qf_itrs'],
            qf_batch_size = default_hyperparameters['qf_batch_size'],

            target_update_period = default_hyperparameters['target_update_period'],
            target_update_tau = default_hyperparameters['target_update_tau'],

            epsilon = default_hyperparameters['epsilon'],

            qf_lr = default_hyperparameters['qf_lr'],
            
            opt_class = optim.Adam
            ):
        super(DQN, self).__init__(env, discount, reward_scale, epochs, rollouts_per_epoch, initial_rollouts, replay_buffer, qf_itrs, qf_batch_size)
        self.qf = qf
        self.target_qf = target_qf

        self.epsilon = epsilon

        self.target_update_period = target_update_period
        self.target_update_tau = target_update_tau

        self.qf_lr = qf_lr
        self.qf_opt = opt_class(self.qf.parameters(), lr=self.qf_lr)

        self.gpu = self.qf.gpu

        self.collector = collector

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

        q_next_act_max = q_next.argmax(dim=1)
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

        if self.current_epoch % self.target_update_period == 0:
            soft_update_from_to(self.qf, self.target_qf, self.target_update_tau)

#        import pdb;pdb.set_trace()
        for idx2 in range(0, self.replay_buffer.n, 8):
#            print(idx2)
            if (self.replay_buffer.buffer['observation'][0, :100] - self.replay_buffer.buffer['observation'][idx2, :100]).abs().sum() != 0:
                break

        init_obs1 = self.replay_buffer.buffer['observation'][0]
        init_obs2 = self.replay_buffer.buffer['observation'][idx2]
        init_q1 = self.qf(init_obs1)[0, 0]
        init_q2 = self.qf(init_obs2)[0, 0]
        out1 = torch.cat([torch.arange(54).unsqueeze(1), init_q1.view(54, 3), init_q2.view(54, 3)], dim=1)

        init_obs21 = self.replay_buffer.buffer['observation'][1]
        init_obs22 = self.replay_buffer.buffer['observation'][1+idx2]
        init_q21 = self.qf(init_obs1)[0, 1]
        init_q22 = self.qf(init_obs2)[0, 1]
        out2 = torch.cat([init_q21.view(54, 3), init_q22.view(54, 3)], dim=1)

        out = torch.cat([init_q1.view(54, 3), init_q21.view(54, 3), init_q2.view(54, 3), init_q22.view(54, 3)], dim=1)

        #Update logs whenever you collect new samples
        self.logger.record_item('Loss', qf_loss, prefix='QF')
        self.logger.record_item('_P1-2 Q Intitial', out, prefix='Performance')
        self.logger.record_item('a) P1 Q Intitial Top 5 1', init_q1.topk(5).indices/3, prefix='Performance')
        self.logger.record_item('c) P1 Q Intitial Top 5 2', init_q2.topk(5).indices/3, prefix='Performance')
        self.logger.record_item('b) P2 Q Intitial Top 5 1', init_q21.topk(5).indices/3, prefix='Performance')
        self.logger.record_item('d) P2 Q Intitial Top 5 2', init_q22.topk(5).indices/3, prefix='Performance')
        
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
