import abc
import time

from catanbot.rl.utils.logging import Logger
from catanbot.rl.utils.rl_utils import compute_returns

class RLAlgorithm(object, metaclass=abc.ABCMeta):
    """
    General template for RL algorithms. We will assume that all RL algorithms implement the following steps:
        1. Collect: run the current policy in the env and return some data to the algorithm to update its networks.
        2. Update: Given some batch of state transitions, update the networks in the algorithm.
        3. Log: Log the results of the update step and optionally save networks.

    Thus, we can think of an epoch in the RL algorithm as an application of collect, update, log.
    """
    @abc.abstractmethod
    def collect(self):
        pass

    @abc.abstractmethod
    def update(self, batch):
        pass

    @abc.abstractmethod
    def log(self):
        pass

    @property
    @abc.abstractmethod
    def hyperparameters(self):
        #return a dict of algorithm hyperparameters and their current values. Note that this makes a copy of the vals so this can't update the vals directly (but can be kwargs to a new instance).
        pass

    @property
    @abc.abstractmethod
    def default_hyperparameters(self):
        #return a dict of algorithm hyperparameters and their default values. Note that this makes a copy of the vals so this can't update the vals directly (but can be kwargs to a new instance).
        pass

    @property
    @abc.abstractmethod
    def networks(self):
        #return a dict of the networks in an algorithm
        pass

class OnPolicyRLAlgorithm(RLAlgorithm):
    """
    Base class for on-policy RL algorithms (A2C, TRPO, PPO, etc.).
    Algorithms of this class should have the following behavior:
        Collect: Run the policy in the env for a number of steps and return it as batch.
        Update: Algorithm-specific.
        Log: Just log the data.
    """
    def __init__(
            self,
            env,
            discount,
            reward_scale,
            epochs,
            steps_per_epoch,
            eval_steps_per_epoch,
            eval_every
            ):
        self.env = env
        self.discount = discount
        self.reward_scale = reward_scale
        self.total_epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.eval_steps_per_epoch = eval_steps_per_epoch
        self.eval_every = eval_every

        self.current_epoch = 0
        self.logger = Logger()

    def collect(self):
        if self.current_epoch % self.eval_every == 0:
            return self.collector.get_rollouts(self.steps_per_epoch), self.eval_collector.get_rollouts(self.eval_steps_per_epoch)
        else:
            return self.collector.get_rollouts(self.steps_per_epoch), self.eval_collector.get_rollouts(0) 

    def log(self):
        self.logger.print_data()

    def train_iteration(self):
        self.current_epoch += 1

        t = time.time()
        trajs, eval_trajs = self.collect()
        collect_time = time.time() - t

        t = time.time()
        self.update(trajs)
        update_time = time.time() - t

        self.logger.record_item("Epoch", self.current_epoch)
        self.logger.record_item("Reward Scale", self.reward_scale)
        self.logger.record_item("Num Episodes", self.logger.get(prefix='', field='Num Episodes', default=0) + trajs['terminal'].float().sum().numpy())
        self.logger.record_item("Num Added Episodes", trajs['terminal'].float().sum().numpy())
        self.logger.record_item("Total Steps", self.logger.get(prefix='', field='Total Steps', default=0) + trajs['observation'].shape[0])
        self.logger.record_item("Return", trajs['reward'].sum(dim=0), prefix = 'Performance') #Sparse-reward assumption
        self.logger.record_item('Collect time', collect_time, prefix = 'Timing')
        self.logger.record_item('Update time', update_time, prefix = 'Timing')

        if self.current_epoch % self.eval_every == 0:
            self.logger.record_item("Eval Return", eval_trajs['reward'].sum(dim=0), prefix = 'Performance') #Sparse-reward assumption
            self.logger.record_item('1 Policy 1 Eval Actions', eval_trajs['action'][0::8].argmax(dim=1)/3, prefix='Performance')
            self.logger.record_item('2 Policy 2 Eval Actions', eval_trajs['action'][1::8].argmax(dim=1)/3, prefix='Performance')
            self.logger.record_item('3 Policy 3 Eval Actions', eval_trajs['action'][2::8].argmax(dim=1)/3, prefix='Performance')
            self.logger.record_item('4 Policy 4 Eval Actions', eval_trajs['action'][3::8].argmax(dim=1)/3, prefix='Performance')
            self.logger.record_item('8 Policy 1 Eval Actions', eval_trajs['action'][7::8].argmax(dim=1)/3, prefix='Performance')
            self.logger.record_item('7 Policy 2 Eval Actions', eval_trajs['action'][6::8].argmax(dim=1)/3, prefix='Performance')
            self.logger.record_item('6 Policy 3 Eval Actions', eval_trajs['action'][5::8].argmax(dim=1)/3, prefix='Performance')
            self.logger.record_item('5 Policy 4 Eval Actions', eval_trajs['action'][4::8].argmax(dim=1)/3, prefix='Performance')

        self.log()

class OffPolicyRLAlgorithm(RLAlgorithm):
    """
    Base class for off-policy RL algorithms (DQN, DDPG, SAC, etc.).
    Algorithms of this class should have the following behavior:
        Collect: Collect trajectories in the environment and store it in the replay buffer.  
        Update: Algorithm-specific.
        Log: Just log the data.
    """
    def __init__(
            self,
            env,
            discount,
            reward_scale,
            epochs,
            steps_per_epoch,
            initial_steps,
            replay_buffer,
            qf_itrs,
            qf_batch_size,
            ):
        self.env = env
        self.discount = discount
        self.reward_scale = reward_scale
        self.total_epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.initial_steps = initial_steps

        self.qf_itrs = qf_itrs
        self.replay_buffer = replay_buffer
        self.qf_batch_size = qf_batch_size

        self.current_epoch = 0
        self.logger = Logger()

    def collect(self):
        return self.replay_buffer.sample(nsamples = self.qf_batch_size)

    def log(self):
        self.logger.print_data()

    def train_iteration(self):
        self.current_epoch += 1
        t = time.time()
        trajs = self.collector.get_rollouts(self.steps_per_epoch)
        collect_time = time.time() - t
        self.replay_buffer.insert(trajs)
        self.logger.record_item("Epoch", self.current_epoch)
        self.logger.record_item("Return", trajs['reward'].sum(dim=0), prefix = 'Performance') #Assuming sparse reward for this domain
        self.logger.record_item("Reward Scale", self.reward_scale)
        self.logger.record_item("Num Episodes", self.logger.get(prefix='', field='Num Episodes', default=0) + trajs['terminal'].float().sum().item())
        self.logger.record_item("Num Added Episodes", trajs['terminal'].float().sum().item())
        self.logger.record_item("Total Steps", self.logger.get(prefix='', field='Total Steps', default=0) + trajs['observation'].shape[0])
        self.logger.record_item('1 Policy 1 Eval Actions', trajs['action'][0:100:8].argmax(dim=1)/3, prefix='Performance')
        self.logger.record_item('2 Policy 2 Eval Actions', trajs['action'][1:100:8].argmax(dim=1)/3, prefix='Performance')
        self.logger.record_item('3 Policy 3 Eval Actions', trajs['action'][2:100:8].argmax(dim=1)/3, prefix='Performance')
        self.logger.record_item('4 Policy 4 Eval Actions', trajs['action'][3:100:8].argmax(dim=1)/3, prefix='Performance')

        for qi in range(self.qf_itrs):
            batch = self.collect()

            t = time.time()
            self.update(batch)
            update_time = time.time() - t

            self.logger.record_item("QF Itr", qi)
            self.logger.record_item('Collect time', collect_time, prefix = 'Timing')
            self.logger.record_item('Update time', update_time, prefix = 'Timing')

            t = time.time()
            self.log()
            log_time = time.time() - t
            self.logger.record_item('Log Time', log_time, prefix= 'Timing')
