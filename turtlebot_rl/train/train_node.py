import os
import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from turtlebot_rl.env.env_node import GazeboEnv
from turtlebot_rl.td3_agent.td3 import TD3
from turtlebot_rl.td3_agent.replay_buffer import ReplayBuffer

SAVE_DIR = os.path.expanduser("/home/artem/ros2_ws/src/turtlebot_rl/weights")
TARGETS = ((-4.0, 4.0), (4.0, 4.0), (4.0, -4.0), (-4.0, -4.0))

# Фаза 1 — пустая комната:
#LOAD_WEIGHTS = None
LOAD_BUFFER  = None
NOISE_START  = 0.5

LOAD_WEIGHTS = '/home/artem/ros2_ws/src/turtlebot_rl/weights/best'         
#LOAD_BUFFER  = '/home/artem/ros2_ws/src/turtlebot_rl/weights/buffer_phase1'          

#NOISE_START  = 0.2           # 0.5 для фазы 1, 0.2 для фазы 2


class TrainNode(Node):
    def __init__(self):
        super().__init__('td3_train_node')
        os.makedirs(SAVE_DIR, exist_ok=True)

        self.env = GazeboEnv(self, goal_positions=TARGETS)

        self.state_dim  = 24
        self.action_dim = 2
        self.max_action = 0.4

        self.agent = TD3(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
        )
        if LOAD_WEIGHTS:
            self.agent.load(os.path.expanduser(LOAD_WEIGHTS))
            self.get_logger().info(f"Loaded weights: {LOAD_WEIGHTS}")

        self.buffer = ReplayBuffer(capacity=500_000)
        self.buffer.init(self.state_dim, self.action_dim)
        if LOAD_BUFFER:
            self.buffer.load(os.path.expanduser(LOAD_BUFFER),
                             keep_existing=False)
            self.get_logger().info(
                f"Loaded buffer: {LOAD_BUFFER}  size={self.buffer.size}"
            )

        self.batch_size        = 256
        self.warmup_steps      = 500 if LOAD_BUFFER else 1000
        self.train_freq        = 2
        self.max_ep_steps      = 500
        self.noise_start       = NOISE_START
        self.noise_end         = 0.05
        self.noise_decay_steps = 50_000
        self.best_reward       = -1e9
        self.save_every        = 10

        self.buffer_save_every = 50

        # ── Состояние ─────────────────────────────────────────────────
        self.state          = self.env.get_state()
        self.episode_reward = 0.0
        self.episode        = 0
        self.ep_steps       = 0
        self.total_steps    = 0
        self.done           = False
        self._state_initialized = False

        self.timer = self.create_timer(0.1, self.loop)
        self.get_logger().info(
            f"TD3 TrainNode started  "
            f"weights={'loaded' if LOAD_WEIGHTS else 'fresh'}  "
            f"buffer={self.buffer.size} transitions"
        )

    def _noise_std(self) -> float:
        frac = min(self.total_steps / self.noise_decay_steps, 1.0)
        return self.noise_start + frac * (self.noise_end - self.noise_start)

    def act(self, state: np.ndarray) -> np.ndarray:
        action = self.agent.select_action(state)
        noise  = np.random.normal(0, self._noise_std(), size=self.action_dim)
        action = np.clip(action + noise, -self.max_action, self.max_action)
        return action

    def _on_episode_end(self):
        self.get_logger().warn(
            f"EP {self.episode:4d} | steps={self.ep_steps:4d} "
            f"| reward={self.episode_reward:8.2f} "
            f"| noise={self._noise_std():.3f} "
            f"| buf={self.buffer.size}"
        )

        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
            self.agent.save(os.path.join(SAVE_DIR, "best"))
            self.get_logger().info(f"  ★ NEW BEST = {self.best_reward:.2f}")

        if self.episode % self.save_every == 0:
            self.agent.save(os.path.join(SAVE_DIR, f"ep_{self.episode:05d}"))

        if self.episode % self.buffer_save_every == 0 and self.episode > 0:
            self.buffer.save(os.path.join(SAVE_DIR, "buffer_latest"))
            self.get_logger().info(
                f"  Buffer saved ({self.buffer.size} transitions)"
            )

        self.episode_reward = 0.0
        self.ep_steps       = 0
        self.episode       += 1
        self.done           = False
        self.state          = self.env.reset()

    def loop(self):
        if not self._state_initialized:
            if self.env.scan is None or self.env.odom is None:
                return
            self.state = self.env.get_state()
            self._state_initialized = True
            return

        if self.done or self.ep_steps >= self.max_ep_steps:
            if self.ep_steps >= self.max_ep_steps:
                self.get_logger().warn("MAX STEPS — forced reset")
            self._on_episode_end()
            return

        action = self.act(self.state)
        next_state, reward, done = self.env.step(action)

        self.buffer.add(self.state, action, reward, next_state, float(done))
        self.state           = next_state
        self.done            = done
        self.episode_reward += reward
        self.total_steps    += 1
        self.ep_steps       += 1

        if (self.buffer.size >= self.warmup_steps
                and self.total_steps % self.train_freq == 0):
            self.agent.train(self.buffer, self.batch_size)

        if self.total_steps % 200 == 0:
            self.get_logger().info(
                f"steps={self.total_steps} buf={self.buffer.size} "
                f"noise={self._noise_std():.3f}"
            )


def main():
    rclpy.init()
    node = TrainNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Saving final model and buffer ...")
        node.agent.save(os.path.join(SAVE_DIR, "final"))
        phase = "phase2" if LOAD_WEIGHTS else "phase1"
        node.buffer.save(os.path.join(SAVE_DIR, f"buffer_{phase}"))
        node.get_logger().info(f"Saved buffer_{phase}.npz")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
