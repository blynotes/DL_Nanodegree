import numpy as np
from physics_sim import PhysicsSim

# Adjusted the reward.


class Task:
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(
        self,
        init_pose=None,
        init_velocities=None,
        init_angle_velocities=None,
        runtime=5.0,
        target_pos=None,
    ):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(
            init_pose, init_velocities, init_angle_velocities, runtime
        )
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = (
            target_pos if target_pos is not None else np.array([0.0, 0.0, 10.0])
        )

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # If far away from the target position, we subtract by scale of scale_factor.
        # scale_factor = 0.3  # This says within 3.333 distance results in positive reward.
        scale_factor = 0.8  # This says within 1.25 distance results in positive reward.

        # Position is a much more important factor, so it has higher scale than velocity.
        # The reward defines what it deems the "best position" is, so we want to make
        # sure that the best position is mostly based on the physical location and now
        # using a really small velocity.
        scale_factor_vel = 0.001

        # Prevent the body velocity of the quadcopter from going too high.
        # This should make it smoother.
        reward = (
            1.0
            - scale_factor * ((abs(self.sim.pose[:3] - self.target_pos)).sum() ** 2)
            - scale_factor_vel * (self.sim.find_body_velocity().sum() ** 2)
            # - scale_factor_vel * abs(self.sim.find_body_velocity().sum())
        )

        # # My task is moving from (0,0,0) to (0,10,0). We want to prevent the
        # # quadcopter from going too high in the air. We can simply subtract
        # # the sigmoid of the z position. If close to the ground, we will subtract close to 0.
        # # If too high it will subtract 1. Divide z position by 3 to allow for some height.
        # # Subtract 0.5 and multiply total by 2 to get values between 0 and 1
        # reward = (
        #     1.0
        #     - scale_factor * (abs(self.sim.pose[:3] - self.target_pos)).sum()
        #     - 2 * (Task.sigmoid(self.sim.pose[2] / 3) - 0.5)
        # )
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(
                rotor_speeds
            )  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
