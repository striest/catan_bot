import time
import copy
import matplotlib.pyplot as plt
import numpy as np;np.set_printoptions(precision=4, linewidth=1e6, suppress=True)
import torch
import ray
import argparse

from catanbot.core.board import Board
from catanbot.agents.independent_actions_agents import IndependentActionsHeuristicAgent, IndependentActionsAgent
from catanbot.board_placement.agents.base import InitialPlacementAgent, MakeDeterministic
from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.board_placement.initial_placement_simulator import InitialPlacementSimulator, InitialPlacementSimulatorWithPenalty
from catanbot.rl.collectors.initial_placement_collector import InitialPlacementCollector

"""
Revised version of MCTS with a learned Q function. Since the slow part is the rollout collection and torch batches by default, no need to parallelize.
Still adopt the model of: search tree to get tasks, run tasks in parallel.
Also, we have two ways of collecting rollouts in non-terminal states - random acts to terminal state, then use Q, or just use Q directly.
Also, combine settlement and road placement into the same tree level.
"""

class MCTSQFNode:
    """
    General node class for the QF MCTS.
    """
    def __init__(self, simulator, parent, prev_act):
        self.simulator = simulator
        self.parent = parent
        self.stats = torch.zeros(4)
        self.depth = 0 if parent is None else parent.depth + 1
        self.prev_act = prev_act
        self.children = []
        #Hooray for garbage code!
        c = InitialPlacementCollector(self.simulator)
        self.flatten_obs = lambda x:c.flatten_observation(x)
        self.visit_count = 0
        self.prob = 1. #Alphago uses policy output to bias exploration. We can use Boltzmann policy.

    @property
    def is_leaf(self):
        return not self.children

    @property
    def is_visited(self):
        return self.stats.sum() > 0

    def expand(self, qf):
        """
        Generate the board states from all possible actions from current player in current state
        """
        obs = self.flatten_obs(self.simulator.observation)
        obs = torch.tensor(obs).float().unsqueeze(0)
        with torch.no_grad():
            qvals = qf(obs).squeeze()
        qvals = qvals[self.simulator.player_idx]
        q_logits = qvals.exp()
        q_probs = q_logits / q_logits.sum()

        act_avail = self.simulator.action_mask().flatten()
        ch = []
        for i in range(self.simulator.action_space['total']):
            if not act_avail[i]:
                continue
            act = np.zeros(self.simulator.action_space['total'])
            act[i] = 1.

            sim_new = copy.deepcopy(self.simulator)
            sim_new.step({'placement':np.reshape(act, [54, 3])})
            new_node = MCTSQFNode(sim_new, self, act, qf)
            new_node.prob = q_probs[i]
            ch.append(new_node)

        return ch

    def rollout(self, qf, lam = 0.5):
        sim_copy = copy.deepcopy(self.simulator)
        while not self.simulator.terminal:
            act_mask = self.simulator.action_mask().flatten()
            acts = np.argwhere(act_mask).flatten()
            act_i = np.random.choice(acts)
            act = np.zeros(self.simulator.action_space['total'])
            act[act_i]=1.
            self.simulator.step({'placement':np.reshape(act, [54, 3])})

        rew = self.simulator.reward()

        obs = self.flatten_obs(sim_copy.observation)
        obs = torch.tensor(obs).float().unsqueeze(0)
        with torch.no_grad():
            qvals = qf(obs).squeeze()

        qmax = qvals.max(dim=1)[0]
#        qmax /= qmax.max() #Normalize Q so everyone gets a win

        self.simulator = sim_copy

        out = lam * qmax + (1-lam) * rew

#        print('Q   = {}'.format(qmax))
#        print('sim = {}'.format(rew))

        return qmax

    def children_ucb(self, c=1.0, bias=0.0):
#        import pdb;pdb.set_trace()
        ucbs = torch.zeros(len(self.children))
        for idx, ch in enumerate(self.children):
            if ch.stats.sum() == 0:
                ucbs[idx] = 1e10
            else:
                ch_wins = ch.stats[self.simulator.player_idx]
                ch_total = ch.stats.sum()
                par_total = self.stats.sum()
                ucbs[idx] = (ch_wins/ch_total) + c*np.sqrt(np.log(par_total + bias)/(ch_total + bias))
        return ucbs        
 
    def __repr__(self, c=1.0):
        return 'state = {}, act = {}, stats = {}, turn = {}({}), visits = {}, ucbs = {}'.format(self.simulator, np.argmax(self.prev_act), self.stats, self.depth, 'RGBY'[self.simulator.player_idx], self.visit_count, self.children_ucb(c=c))

class MCTSQFSearch:
    """
    Basically the same as the other one.
    """
    def __init__(self, simulator, qf):
        self.simulator = simulator
        self.original_board = simulator.simulator.board
        self.original_agents = simulator.players
        self.root = MCTSQFNode(copy.deepcopy(simulator), None, None, qf)
        self.qf = qf

    def search(self, n_rollouts = None, max_time = None, c=1.0, verbose=False):
        """
        1. Search for a leaf node
        2. If the leaf hasn't been visited, collect a rollout and propagate up.
        3. If the leaf has been visited, expand it
        """
        assert not (n_rollouts is None and max_time is None), 'Need to set one of rollouts or time'
        if n_rollouts is None:
            n_rollouts = float('inf')
        if max_time is None:
            max_time = float('inf')

        prev = time.time()
        r = 0
        t_running = 0
        r_t_running = 0

        ptime = 0
        rtime = 0
        while r < n_rollouts and t_running < max_time:
            """
            if r > 150:
                import pdb;pdb.set_trace()
                print(self.root.children_ucb(c=0.))
                print(self.root.children_ucb(c=0.1))
                print(self.root.children_ucb(c=0.5))
                print(self.root.children_ucb(c=1.0))
            """
        #    print(self)
            t_itr = time.time()-prev
            t_remaining =  (t_running / (r+1)) * (n_rollouts - r) if max_time == float('inf') else max_time - t_running
            t_running += t_itr
            prev = time.time()
#            import pdb;pdb.set_trace()
            curr, path = self.find_leaf(c=c)
            maxdepth = curr.simulator.terminal
#            print(curr)
            if verbose:
                print('Rollout #{} (t={:.2f}s) time elapsed = {:.2f}s, time remaining = {:.2f}s rollout time = {:.2f}, place time = {:.2f}, simulate time {:.2f}'.format(r, t_itr, t_running, t_remaining, r_t_running, ptime, rtime))
                print('depth = {}, path = {}'.format(curr.depth, path))
                if maxdepth:
                    print('max depth')
            if curr.is_visited and not maxdepth:
                ch = curr.expand(qf=self.qf)
                curr.children = ch
            else:
                rtstart = time.time()
                result = curr.rollout(qf=self.qf)
                place_time = rollout_time = time.time() - rtstart
                ptime += place_time
                rtime += rollout_time
                r_t_running += (time.time() - rtstart)
                r += 1

                while curr:
                    curr.visit_count += 1
                    curr.stats += result
                    curr = curr.parent

        #Restore the simulator if you need to call MCTS again
#        self.simulator.simulator.reset_from(self.original_board, players = self.original_agents)

    def find_leaf(self, c=1.0):
        """
        find leaf in MCTS tree: take action with highest UCB to get there
        """    
        curr = self.root
        path = []
        while not curr.is_leaf:
            #Randomly sample argmaxes to get better coverage when using multiple trees
            ucbs = curr.children_ucb(c=0.)
            #exploration_bonuses = c * torch.tensor([ch.prob / (1+ch.visit_count) for ch in curr.children])
            exploration_bonuses = c * torch.tensor([ch.prob * (np.sqrt(curr.visit_count) / (1+ch.visit_count)) for ch in curr.children])
            ucbs += exploration_bonuses
            _max = ucbs.max()
            maxidxs = torch.where(ucbs >= _max)[0]
            idx = maxidxs[torch.randint(0, maxidxs.shape[0], (1,))]
            curr = curr.children[idx]
            path.append([curr.prev_act.argmax()//3, curr.prev_act.argmax()%3])
        return curr, path

    def get_optimal_path(self):
        path = [self.root]
        curr = self.root
        while not curr.is_leaf:
            ucbs = curr.children_ucb(c=0)
            idx = ucbs.argmax()
            curr = curr.children[idx]
            path.append(curr)
        return path

    def get_top_k(self, k=5):
        """
        Get the top k placements (first settlement and road)
        """
        curr = self.root
        options = []
        ucbs = curr.children_ucb(c=0)
        for i, ch in enumerate(curr.children):
            a_s = ch.prev_act.argmax()//3
            a_r = ch.prev_act.argmax()%3
            options.append((ch.stats, [a_s, a_r], ucbs[i]))

        options.sort(key=lambda x:x[2], reverse=True)
        return options[:k]

    def __repr__(self):
        return self.dump(self.root, 0)

    def dump(self, node, depth, c=0.):
        out = '\t' * depth
        out += node.__repr__(c)
        out += '\n'
        for i, ch in enumerate(node.children):
            out += str(depth+1) + ':' + str(i) + self.dump(ch, depth +1, c)
        return out

class RayMCTSQFSearch:
    """
    Use Ray to parallelize the search.
    """
    def __init__(self, simulator, qf, n_threads = 1):
        ray.init(ignore_reinit_error=True)
        self.simulator = simulator
        self.original_board = simulator.simulator.board
        self.original_agents = simulator.players
        self.root = MCTSQFNode(copy.deepcopy(simulator), None, None)
        self.qf = qf
        self.n_threads = n_threads
        self.workers = []
        self.sigstop = False
        for _ in range(self.n_threads):
            self.workers.append(MCTSQFRayWorker.remote(qf))

    def search(self, n_rollouts = None, max_time = None, c=1.0, verbose=False):
        """
        1. Search for a leaf node
        2. If the leaf hasn't been visited, collect a rollout and propagate up.
        3. If the leaf has been visited, expand it
        """
        assert not (n_rollouts is None and max_time is None), 'Need to set one of rollouts or time'
        if n_rollouts is None:
            n_rollouts = float('inf')
        if max_time is None:
            max_time = float('inf')

        prev = time.time()
        r = 0
        t_running = 0
        parallel_time = 0
        r_t_running = 0
        while r < n_rollouts and t_running < max_time and not self.sigstop:
#            import pdb;pdb.set_trace()
        #    print(self)

            t_itr = time.time()-prev
            t_remaining =  (t_running / (r+1)) * (n_rollouts - r) if max_time == float('inf') else max_time - t_running
            t_running += t_itr
            prev = time.time()
            leaves, paths = self.find_leaves(c=c, verbose=verbose)
            explored_paths = set()
            for i in range(len(paths)):
                if str(paths[i]) in explored_paths:
                    print('Random path')
                    leaf, path = self.find_random_leaf()
                    leaves[i] = leaf
                    paths[i] = path

                explored_paths.add(str(paths[i]))
            
            tasks = []
            taskids = []

            #Generate expansions/rollouts. Note that we have to detach the leaf from the tree to keep Ray from copying the tree. Reattach after the rollouts are collected
            widx = 0
            for leaf, path in zip(leaves, paths):
                maxdepth = leaf.simulator.terminal
#                print(curr)
                leafcopy = copy.copy(leaf)
                leafcopy.parent = None
                if leaf.is_visited and not maxdepth:
                    tasks.append(self.workers[widx].expand.remote(leafcopy))
                    taskids.append('e')
                else:
                    tasks.append(self.workers[widx].rollout.remote(leafcopy))
                    taskids.append('r')
                widx += 1

            p_time = time.time()
            tasks = ray.get(tasks)
            parallel_time += time.time() - p_time

            #Put expansions/rollouts into the tree
            for leaf, task, taskid in zip(leaves, tasks, taskids):
                if taskid == 'e':
                    leaf.children = task
                    print('Expanded {} nodes.'.format(len(task)))
                    #Idk why, but reset the parent for the children (the parent is probably copied when it moves to the Ray actor)
                    for ch in leaf.children:
                        ch.parent = leaf
                else:
                    r += 1
                    while leaf:
                        leaf.stats = leaf.stats + task
                        leaf = leaf.parent

            if verbose:
                print('Rollout #{} (t={:.2f}s) time elapsed = {:.2f}s, time remaining = {:.2f}s rollout/expand time = {:.2f}'.format(r, t_itr, t_running, t_remaining, parallel_time))
                best = self.get_optimal_path()
                best = [np.argmax(n.prev_act) for n in best[1:]]
                print('best = {}'.format(best))
                placements = [path for path in paths]
                placements.sort(key=lambda x:x[0] * 100 + x[1] if len(x) >1 else 0)
                print('explored {}'.format(placements))
                print('avg depth = {}'.format(np.array([len(p) for p in paths]).mean()))

        #Restore the simulator if you need to call MCTS again
#        self.simulator.simulator.reset_from(self.original_board, players = self.original_agents)

    def find_leaves(self, c=1.0, verbose=False):
        leaves = []
        paths = []
        path_unique = set()
        for i in range(self.n_threads):
            leaf, path = self.find_leaf(c=c, bias=i)
            leaves.append(leaf)
            paths.append(path)
            pathstr = ' '.join([str(v) for v in path])
            path_unique.add(pathstr)
        if verbose:
            print('{} unique/{} total'.format(len(path_unique), self.n_threads))
        return leaves, paths

    def find_leaf(self, c=1.0, bias=0.):
        """
        find leaf in MCTS tree: take action with highest UCB to get there
        """    
        curr = self.root
        path = []
        while not curr.is_leaf:
            curr.visit_count += 1
            #Randomly sample argmaxes to get better coverage when using multiple trees
            ucbs = curr.children_ucb(c=0., bias=bias)
            exploration_bonuses = c * torch.tensor([ch.prob * (np.sqrt(curr.visit_count) / (1+ch.visit_count)) for ch in curr.children])
#            print('ucb=', ucbs)
#            print('ebs=', exploration_bonuses)
            ucbs += exploration_bonuses
            _max = ucbs.max()
            maxidxs = torch.where(ucbs >= _max)[0]
            idx = maxidxs[torch.randint(0, maxidxs.shape[0], (1,))]
            curr = curr.children[idx]
#            path.append([curr.prev_act.argmax()//3, curr.prev_act.argmax()%3])
            path.append(curr.prev_act.argmax())
        return curr, path

    def find_random_leaf(self):
        """
        Included to not waste expansion time.
        """
        curr = self.root
        path = []
        while not curr.is_leaf:
#            curr.visit_count += 1
            curr = np.random.choice(curr.children)
            path.append(curr.prev_act.argmax())
        return curr, path
        

    def get_optimal_path(self):
        path = [self.root]
        curr = self.root
        while not curr.is_leaf:
            ucbs = curr.children_ucb(c=0)

            visit_counts = torch.tensor([ch.visit_count for ch in curr.children])

            idx = ucbs.argmax()
#            idx = visit_counts.argmax()
            curr = curr.children[idx]
            path.append(curr)
        return path

    def get_top_k(self, k=5):
        """
        Get the top k placements (first settlement and road)
        """
        curr = self.root
        options = []
        ucbs = curr.children_ucb(c=0)
        visit_counts = torch.tensor([ch.visit_count for ch in curr.children])
        for i, ch in enumerate(curr.children):
            a_s = ch.prev_act.argmax()//3
            a_r = ch.prev_act.argmax()%3
            options.append((ch, [a_s, a_r], ucbs[i]))

        options.sort(key=lambda x:x[2], reverse=True)
        return options[:k]

    def __repr__(self):
        return self.dump(self.root, 0)

    def dump(self, node, depth, c=1.0):
        out = '\t' * depth
        out += node.__repr__(c)
        out += '\n'
        for i, ch in enumerate(node.children):
            out += str(depth+1) + ':' + str(i) + self.dump(ch, depth +1, c)
        return out

@ray.remote
class MCTSQFRayWorker:
    def __init__(self, qf):
        self.qf = qf

    def expand(self, x):
        """
        Generate the board states from all possible actions from current player in current state
        """
        node = copy.deepcopy(x)
        obs = node.flatten_obs(node.simulator.observation)
        obs = torch.tensor(obs).float().unsqueeze(0)
        with torch.no_grad():
            qvals = self.qf(obs).squeeze()
        qvals = qvals[node.simulator.player_idx]
        q_logits = qvals.exp()
        q_probs = q_logits / q_logits.sum()

        act_avail = node.simulator.action_mask().flatten()

        prod = node.simulator.simulator.board.compute_production()[:, 1]
        prod = np.repeat(prod, 3)

        ch = []
        for i in range(node.simulator.action_space['total']):
            if not act_avail[i] or prod[i] < 7:
                continue
            act = np.zeros(node.simulator.action_space['total'])
            act[i] = 1.

            sim_new = copy.deepcopy(node.simulator)
            sim_new.step({'placement':np.reshape(act, [54, 3])})
            new_node = MCTSQFNode(sim_new, node, act)
            new_node.prob = q_probs[i]
            ch.append(new_node)

        return ch
        
    def rollout(self, x, lam=0.25):
        node = copy.deepcopy(x)
        sim_copy = copy.deepcopy(node.simulator)
        while not node.simulator.terminal:
            act_mask = node.simulator.action_mask().flatten()
            acts = np.argwhere(act_mask).flatten()
            act_i = np.random.choice(acts)
            act = np.zeros(node.simulator.action_space['total'])
            act[act_i]=1.
            node.simulator.step({'placement':np.reshape(act, [54, 3])})

        rew = node.simulator.reward()

#        obs = node.flatten_obs(sim_copy.observation)
        obs = node.flatten_obs(node.simulator.observation)
        obs = torch.tensor(obs).float().unsqueeze(0)
        with torch.no_grad():
            qvals = self.qf(obs).squeeze()

        qmax = qvals.max(dim=1)[0]
        qmax /= qmax.sum()

        node.simulator = sim_copy

        out = lam * qmax + (1-lam) * rew

        return qmax

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse videomaker params')
    parser.add_argument('--qf_fp', type=str, required=True, help='location to the Q network')
    args = parser.parse_args()
    qf = torch.load(args.qf_fp)
    qf.eval()

    b = Board() 
    b.reset()
    agents = [IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b)]
    s = IndependentActionsCatanSimulator(board=b, players = agents, max_vp=10)

    placement_agents = [InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b)]

    placement_simulator = InitialPlacementSimulator(s, placement_agents)

    collector = InitialPlacementCollector(placement_simulator, reset_board=True, reward_scale=1.)

    mcts = RayMCTSQFSearch(placement_simulator, qf, n_threads=12)
    import pdb;pdb.set_trace()
#    mcts = MCTSQFSearch(placement_simulator, qf)
    mcts.search(max_time=600.0, verbose=True, c=2.0)
#    print(mcts)
    print(mcts.get_optimal_path())
    print(mcts.get_top_k())
    placement_simulator.render()
