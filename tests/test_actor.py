import numpy as np
import torch

from turtlebot_rl.env.env_node import GazeboEnv
from turtlebot_rl.td3_agent.actor import Actor


class _DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass


class _DummyPublisher:
    def publish(self, *args, **kwargs):
        pass


class _DummyClient:
    def wait_for_service(self, timeout_sec=1.0):
        return True

    def call_async(self, req):
        return None


class _DummyNode:
    def create_subscription(self, *args, **kwargs):
        return None

    def create_publisher(self, *args, **kwargs):
        return _DummyPublisher()

    def create_client(self, *args, **kwargs):
        return _DummyClient()

    def get_logger(self):
        return _DummyLogger()


def test_actor_shape_and_action_bounds():
    model = Actor(state_dim=24, action_dim=2, max_action=0.4)
    states = torch.zeros((8, 24), dtype=torch.float32)

    actions = model(states)

    assert actions.shape == (8, 2)
    values = actions.detach().cpu().numpy()
    assert np.all(values[:, 0] >= -0.4)
    assert np.all(values[:, 0] <= 0.4)
    assert np.all(values[:, 1] >= -0.4)
    assert np.all(values[:, 1] <= 0.4)


def test_wall_contact_reward_is_negative():
    env = GazeboEnv(_DummyNode(), goal_x=4.0, goal_y=4.0)
    env._teleport = lambda *args, **kwargs: None
    env._steps_since_reset = 10
    env._prev_dist = 2.0
    env.get_state = lambda: np.concatenate([
        np.full(20, 0.1, dtype=np.float32),
        np.array([2.0 / 10.0, 0.0, 1.0, 0.1], dtype=np.float32),
    ])

    _, reward, done = env.step(np.array([0.2, 0.0], dtype=np.float32))

    assert done is False
    assert reward < 0.0


def test_lidar_reduction_preserves_obstacles_between_sampled_rays():
    env = GazeboEnv(_DummyNode())
    ranges = np.full(360, env.LIDAR_MAX, dtype=np.float32)
    ranges[5] = 0.12

    reduced = env._reduce_lidar(ranges)

    assert reduced.shape == (env.LIDAR_N,)
    assert np.isclose(np.min(reduced), 0.12)


def test_realistic_forward_progress_has_positive_reward():
    env = GazeboEnv(_DummyNode(), goal_x=4.0, goal_y=4.0)
    env._teleport = lambda *args, **kwargs: None
    env._steps_since_reset = 1
    env._prev_dist = 3.0
    env.get_state = lambda: np.concatenate([
        np.ones(20, dtype=np.float32),
        np.array([2.95 / 10.0, 0.0, 1.0, 1.0], dtype=np.float32),
    ])

    _, reward, done = env.step(np.array([0.2, 0.0], dtype=np.float32))

    assert done is False
    assert reward > 0.0


def test_target_selection_avoids_repeating_current_goal():
    targets = ((-4.0, 4.0), (4.0, 4.0), (4.0, -4.0))
    env = GazeboEnv(_DummyNode(), goal_positions=targets)
    moved_markers = []
    env._set_goal_marker = lambda x, y: moved_markers.append((x, y))

    env._select_goal()

    assert (env.goal_x, env.goal_y) in targets
    assert (env.goal_x, env.goal_y) != targets[0]
    assert moved_markers == [(env.goal_x, env.goal_y)]
