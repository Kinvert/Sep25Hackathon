import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.drone_pp import binding

class DronePP(pufferlib.PufferEnv):
    def __init__(
        self,
        # Training parameters (don't affect physics)
        num_envs=24,  # Can train with many envs
        num_drones=64,  # Can train with many drones per env

        # ALL PHYSICS MUST MATCH ISAAC SIM EXACTLY
        # These are ignored - using hardcoded Isaac Sim values in C
        max_rings=5,
        reward_min_dist=1.6,
        reward_max_dist=77.0,
        dist_decay=0.15,
        w_position=1.13,
        w_velocity=0.15,
        w_stability=2.0,
        w_approach=2.2,
        w_hover=1.5,
        pos_const=0.63,
        pos_penalty=0.03,
        grip_k_min=1.0,
        grip_k_max=15.0,
        grip_k_decay=0.095,

        # Accept any config params to avoid errors
        box_base_density=None,
        box_k_growth=None,
        reward_grip=None,
        reward_ho_drop=None,
        reward_hover=None,

        render_mode=None,
        report_interval=1024,
        buf=None,
        seed=0,
        **kwargs,  # Catch any other config params
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(42,),
            dtype=np.float32,
        )

        # Changed from 4 motor commands to 2 velocity commands (vx, vy)
        # PID controller handles altitude/attitude automatically
        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

        self.num_agents = num_envs*num_drones
        self.render_mode = render_mode
        self.report_interval = report_interval
        self.tick = 0

        super().__init__(buf)
        self.actions = self.actions.astype(np.float32)

        c_envs = []
        for i in range(num_envs):
            # Actions are now 2D velocity commands per agent (not 4 motor commands)
            # Isaac Sim compatible - minimal parameters only
            c_envs.append(binding.env_init(
                self.observations[i*num_drones:(i+1)*num_drones],
                self.actions[i*num_drones:(i+1)*num_drones],  # PufferEnv already allocates correct size
                self.rewards[i*num_drones:(i+1)*num_drones],
                self.terminals[i*num_drones:(i+1)*num_drones],
                self.truncations[i*num_drones:(i+1)*num_drones],
                i,
                num_agents=num_drones,
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
