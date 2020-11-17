import torch

from catanbot.rl.replaybuffers.base import ReplayBuffer

class SimpleReplayBuffer:
    """
    Generic replay buffer with nothing fancy
    Some assumptions:
        1. Reward is a 4-tensor
        2. Env is discrete
    """

    def __init__(self, env, capacity = int(1e7), gpu=False):
        super(SimpleReplayBuffer, self).__init__()

        self.capacity = int(capacity)
        self.obs_dim = env.observation_space['total']
        self.n = 0 #the index to start insering into
        self.gpu = gpu

        self.act_dim = env.action_space['total']
        self.buffer = {
            'observation': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, self.obs_dim),
            'action': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, self.act_dim),
            'reward':torch.tensor([float('inf')], device=self.device).repeat(self.capacity, 4),
            'next_observation': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, self.obs_dim),
            'terminal': torch.tensor([True], device=self.device).repeat(self.capacity, 1),
            'pidx': torch.tensor([-1], device=self.device).repeat(self.capacity, 1)
        }    

    def insert(self, samples):
        """
        Assuming samples are being passed in as a dict of tensors.
        """
        assert len(samples['observation']) == len(samples['action']) == len(samples['reward']) == len(samples['next_observation']) == len(samples['terminal']) == len(samples['pidx']), \
        "expected all elements of samples to have same length, got: {} (\'returns\' should be a different length though)".format([(k, len(samples[k])) for k in samples.keys()])

        nsamples = len(samples['observation'])

        for k in self.buffer.keys():
            for i in range(nsamples):
                self.buffer[k][(self.n + i) % self.capacity] = samples[k][i]

        self.n += nsamples

    def nelements(self):
        return min(self.n, self.capacity)

    def sample(self, nsamples):
        """
        Get a batch of samples from the replay buffer.
        """

        #Don't want to sample placeholders, so min n and capacity.
        idxs = torch.LongTensor(nsamples).random_(0, min(self.n, self.capacity)) 

        out = {k:self.buffer[k][idxs] for k in self.buffer.keys()}
        
        if self.gpu:
            out = {k:out[k].cuda() for k in out.keys()}

        return out

    @property
    def device(self):
        return 'cuda' if self.gpu else 'cpu'

    def __repr__(self):
        return "buffer = {} \nn = {}".format(self.buffer, self.n)

