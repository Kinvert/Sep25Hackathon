'''Drone Pick and Place Environment for the REVEL x LycheeAI Hackathon

Action Space (10 discrete actions):
0: MOVE_FORWARD - Move forward relative to drone heading
1: MOVE_BACKWARD - Move backward relative to drone heading
2: MOVE_LEFT - Strafe left relative to drone heading
3: MOVE_RIGHT - Strafe right relative to drone heading
4: MOVE_UP - Increase altitude
5: MOVE_DOWN - Decrease altitude
6: ROTATE_LEFT - Yaw counter-clockwise
7: ROTATE_RIGHT - Yaw clockwise
8: GRIPPER_OPEN - Open gripper to release objects
9: GRIPPER_CLOSE - Close gripper to grasp objects

Observation Space (45 continuous values):
- Drone state (14): position(3), velocity(3), quaternion(4), angular_velocity(3), gripper_state(1)
- Object states (21): For each of 3 objects - position(3), velocity(3), status(1)
- Target zones (8): For each of 2 targets - position(3), has_object(1)
- Task info (2): time_remaining(1), task_progress(1)
'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.drone_pickplace import binding

class DronePickPlace(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, 
                num_drones=1, num_objects=1, num_targets=1,
                world_size=2.0, max_height=1.5, max_steps=500,
                reward_approach=0.01, reward_complete=1.0, reward_grasp=1.0,
                reward_place=1.0, penalty_crash=-1.0, penalty_no_progress=-0.1,
                penalty_time=-0.001,
                buf=None, seed=0, **kwargs):

        obs_per_drone = 45
        total_obs_size = num_drones * obs_per_drone

        # Single agent wrapper for now (even with multiple drones)
        # Can be extended to multi-agent later todo
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_obs_size,), 
            dtype=np.float32
        )

        # 10 discrete actions for quadcopter control
        # Movement: forward, backward, left, right, up, down (6)
        # Rotation: yaw left, yaw right (2)
        # Gripper: open, close (2)
        self.single_action_space = gymnasium.spaces.Discrete(10)

        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        self.num_drones = num_drones
        self.num_objects = num_objects
        self.num_targets = num_targets
        self.world_size = world_size
        self.max_height = max_height
        self.max_steps = max_steps

        self.reward_approach = reward_approach
        self.reward_complete = reward_complete
        self.reward_grasp = reward_grasp
        self.reward_place = reward_place
        self.penalty_crash = penalty_crash
        self.penalty_no_progress = penalty_no_progress
        self.penalty_time = penalty_time

        super().__init__(buf)

        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed,
            num_drones=num_drones,
            num_objects=num_objects,
            num_targets=num_targets,
            world_size=world_size,
            max_height=max_height,
            max_steps=max_steps,
            reward_approach=reward_approach,
            reward_complete=reward_complete,
            reward_grasp=reward_grasp,
            reward_place=reward_place,
            penalty_crash=penalty_crash,
            penalty_no_progress=penalty_no_progress,
            penalty_time=penalty_time
        )

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1

        if self.num_drones > 1:
            expanded_actions = np.repeat(actions, self.num_drones, axis=0)
            self.actions[:] = expanded_actions
        else:
            self.actions[:] = actions

        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

class DronePickPlaceMultiAgent(DronePickPlace):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128,
                 num_drones=2, num_objects=3, num_targets=2,
                 world_size=2.0, max_height=1.5, max_steps=1000,
                 buf=None, seed=0):

        super().__init__(
            num_envs=num_envs * num_drones,  # Each drone is an agent
            render_mode=render_mode,
            log_interval=log_interval,
            num_drones=1,  # Each "env" is now a single drone
            num_objects=num_objects,
            num_targets=num_targets,
            world_size=world_size,
            max_height=max_height,
            max_steps=max_steps,
            buf=buf,
            seed=seed
        )

        self.num_agents = num_envs * num_drones


def test_environment():
    """Test the environment with random actions"""
    import time

    print("Testing Drone Pick & Place Environment")
    print("=" * 50)

    # Single environment test
    env = DronePickPlace(num_envs=1, render_mode="human")
    obs, _ = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.single_action_space}")
    print(f"Observation space: {env.single_observation_space}")

    steps = 0
    episode_reward = 0

    for _ in range(1000):
        # Random action
        action = np.random.randint(0, 10, size=(1,))

        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward[0]
        steps += 1

        if render_mode == "human":
            env.render()
            time.sleep(0.01)

        if done[0] or truncated[0]:
            print(f"Episode ended. Steps: {steps}, Reward: {episode_reward:.2f}")
            obs, _ = env.reset()
            steps = 0
            episode_reward = 0

    env.close()

    # Performance test with multiple environments
    print("\nPerformance Test")
    print("-" * 50)

    N = 256  # Number of parallel environments
    env = DronePickPlace(num_envs=N)
    env.reset()

    CACHE = 1024
    actions = np.random.randint(0, 10, (CACHE, N))

    i = 0
    start = time.time()
    total_steps = 0

    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        total_steps += N
        i += 1

    elapsed = time.time() - start
    sps = int(total_steps / elapsed)
    print(f'DronePickPlace SPS: {sps:,} ({N} environments)')
    print(f'Steps per environment: {total_steps // N}')

    env.close()


if __name__ == '__main__':
    test_environment()
