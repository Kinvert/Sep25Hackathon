import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.drone_pp import binding

class DronePP(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=16,
        num_drones=64,
        max_rings=5,

        reward_min_dist=2.0,
        reward_max_dist=75.0,
        dist_decay=0.15,

        w_position=1.0,
        w_velocity=0.004,
        w_stability=1.8,
        w_approach=1.7,
        w_hover=1.8,

        pos_const=0.23,
        pos_penalty=0.2,

        grip_k_min=1.0,
        grip_k_max=18.0,
        grip_k_decay=0.12,

        render_mode=None,
        report_interval=2048,
        buf=None,
        seed=0,
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(42,),
            dtype=np.float32,
        )

        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

        self.num_agents = num_envs*num_drones
        self.render_mode = render_mode
        self.report_interval = report_interval
        self.tick = 0

        super().__init__(buf)
        self.actions = self.actions.astype(np.float32)

        c_envs = []
        for i in range(num_envs):
            c_envs.append(binding.env_init(
                self.observations[i*num_drones:(i+1)*num_drones],
                self.actions[i*num_drones:(i+1)*num_drones],
                self.rewards[i*num_drones:(i+1)*num_drones],
                self.terminals[i*num_drones:(i+1)*num_drones],
                self.truncations[i*num_drones:(i+1)*num_drones],
                i,
                env_i=i, #racedb
                num_agents=num_drones,
                max_rings=max_rings,

                reward_min_dist=reward_min_dist,
                reward_max_dist=reward_max_dist,
                dist_decay=dist_decay,

                w_position=w_position,
                w_velocity=w_velocity,
                w_stability=w_stability,
                w_approach=w_approach,
                w_hover=w_hover,

                pos_const=pos_const,
                pos_penalty=pos_penalty,

                grip_k_min=grip_k_min,
                grip_k_max=grip_k_max,
                grip_k_decay=grip_k_decay
            ))

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions

        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            log_data = binding.vec_log(self.c_envs)
            if log_data:
                info.append(log_data)

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    env = DronePP(num_envs=1000)
    env.reset()
    tick = 0

    actions = [env.action_space.sample() for _ in range(atn_cache)]

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {env.num_agents * tick / (time.time() - start)}")

if __name__ == "__main__":
    test_performance()
