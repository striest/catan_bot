import torch
from torch import nn, optim, distributions

from catanbot.rl.algos.base import OffPolicyRLAlgorithm

class VFTrainer(OffPolicyRLAlgorithm):
    """
    Simple RL that trains a V function
    """
    default_hyperparameters = {
                    'vf_itrs': 200,
                    'vf_batch_size': 128,
                    'vf_lr': 1e-3
                    }

    def __init__(
            self,
            env,
            vf,
            replay_buffer,
            collector,

            discount = 0.99,
            reward_scale = 1.0,

            epochs = int(1e6),
            rollouts_per_epoch = 10,

            vf_itrs = default_hyperparameters['vf_itrs'],
            vf_batch_size = default_hyperparameters['vf_batch_size'],

            vf_lr = default_hyperparameters['vf_lr'],
            
            opt_class = optim.Adam
            ):
        super(VFTrainer, self).__init__(env, discount, reward_scale, epochs, rollouts_per_epoch, 0, replay_buffer, vf_itrs, vf_batch_size)
        self.vf = vf

        self.vf_lr = vf_lr
        self.vf_opt = opt_class(self.vf.parameters(), lr=self.vf_lr)
        self.reward_scale = reward_scale

        self.collector = collector
        self.collector.reward_scale = self.reward_scale

    def update(self, batch):
        """
        Use regular Bellman loss to update
        V(s_t+1) = r_t + gamma*V(s_t)
        We could potentially use cross entropy.
        """
        v_next = self.vf(batch['next_observation'])
        v_curr = self.vf(batch['observation'])
        rew = batch['reward']
        term = batch['terminal']

        v_target = (self.discount*v_next*(~term) + rew).detach()
        v_error = v_target - v_curr
        vf_loss = v_error.pow(2).mean()

        self.vf_opt.zero_grad()
        vf_loss.backward()
        self.vf_opt.step()

        #Update logs whenever you collect new samples
        self.logger.record_item('Loss', vf_loss, prefix='VF')
        self.logger.record_item('Prediction', v_curr[0], prefix='VF')

    @property
    def hyperparameters(self):
        return {
                'vf_itrs': self.vf_itrs,
                'vf_batch_size': self.vf_batch_size,
                'vf_lr': self.vf_lr
            }

    @property
    def networks(self):
        return {
                'vf':self.vf,
            }
