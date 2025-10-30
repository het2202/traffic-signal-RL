import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TrafficEnv(gym.Env):
    """Simple 4-way intersection environment.

    State: continuous vector of queue lengths: [north, south, east, west]
    Action: 0 = NS green, 1 = EW green
    Reward: negative sum of queue lengths (minimize waiting)

    Simple car arrival model: Poisson arrivals per lane per timestep.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, max_queue=20, arrival_rate=0.3, seed=None):
        super().__init__()
        self.num_lanes = 4
        self.max_queue = max_queue
        self.arrival_rate = arrival_rate
        self.observation_space = spaces.Box(low=0.0, high=float(max_queue),
                                            shape=(self.num_lanes,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # NS or EW green
        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        # queues: [N, S, E, W]
        self.queues = np.zeros(self.num_lanes, dtype=np.float32)
        self.t = 0
        obs = self.queues.copy()
        return obs, {}

    def step(self, action):
        assert self.action_space.contains(action)
        # cars arrive
        arrivals = self.np_random.poisson(self.arrival_rate, size=self.num_lanes)
        self.queues = np.minimum(self.max_queue, self.queues + arrivals)

        # serve cars based on green direction:
        # if NS green -> lanes 0 and 1 (N,S) get service
        served = np.zeros(self.num_lanes, dtype=np.float32)
        service_rate = 1  # cars cleared per green lane per timestep
        if action == 0:
            served[0] = min(service_rate, self.queues[0])
            served[1] = min(service_rate, self.queues[1])
        else:
            served[2] = min(service_rate, self.queues[2])
            served[3] = min(service_rate, self.queues[3])

        self.queues -= served
        self.queues = np.clip(self.queues, 0, self.max_queue)

        # reward: negative sum of queues (we want to minimize waiting)
        reward = -float(self.queues.sum())

        self.t += 1
        done = False
        info = {'served': served.sum(), 'queues': self.queues.copy()}
        obs = self.queues.copy()
        return obs, reward, done, False, info

    def render(self, mode='human'):
        # Simple textual render
        print(f"t={self.t} queues=[N:{int(self.queues[0])} S:{int(self.queues[1])} E:{int(self.queues[2])} W:{int(self.queues[3])}]")
