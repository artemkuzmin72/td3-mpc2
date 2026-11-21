import math
import threading
import time
import numpy as np

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetEntityState


class GazeboEnv:
    COLLISION_DIST = 0.17
    GRACE_STEPS    = 8
    LIDAR_N        = 20
    LIDAR_MAX      = 3.5
    GOAL_THRESHOLD = 0.5
    PHYSICS_WAIT   = 0.15
    TELEPORT_SLEEP = 0.5
    WALL_DETECT    = 0.65
    WALL_CLEAR     = 0.90
    WALL_MIN_GAIN  = 0.15

    def __init__(self, node: Node, goal_x: float = -4.0, goal_y: float = 4.0,
                 goal_positions: tuple[tuple[float, float], ...] | None = None):
        self.node   = node
        self.goal_x = goal_x
        self.goal_y = goal_y
        self._goal_positions = goal_positions or ((goal_x, goal_y),)
        self._goal_rng = np.random.default_rng()

        self.scan: np.ndarray | None = None
        self.odom = None
        self._sensor_lock = threading.Lock()
        self._sensor_group = ReentrantCallbackGroup()
        self._steps_since_reset = 0
        self._prev_dist = math.sqrt(goal_x**2 + goal_y**2)
        self._wall_encounter_active = False
        self._wall_encounter_dist = None

        self.node.create_subscription(
            LaserScan, '/scan', self._scan_cb, 10,
            callback_group=self._sensor_group)
        self.node.create_subscription(
            Odometry, '/odom', self._odom_cb, 10,
            callback_group=self._sensor_group)
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)

        self._teleport_client = self.node.create_client(
            SetEntityState, '/set_entity_state')

        while not self._teleport_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("Waiting for /set_entity_state ...")

        self.node.get_logger().info("GazeboEnv READY")
        self._teleport(0.0, 0.0)

    def _scan_cb(self, msg: LaserScan):
        r = np.array(msg.ranges, dtype=np.float32)
        r[~np.isfinite(r)] = self.LIDAR_MAX
        with self._sensor_lock:
            self.scan = r

    def _odom_cb(self, msg: Odometry):
        with self._sensor_lock:
            self.odom = msg

    def reset(self) -> np.ndarray:
        self._select_goal()
        self._teleport(0.0, 0.0)
        self._steps_since_reset = 0
        return self.get_state()

    def get_state(self) -> np.ndarray:
        with self._sensor_lock:
            scan = None if self.scan is None else self.scan.copy()
            odom = self.odom

        if scan is None or odom is None:
            return np.zeros(24, dtype=np.float32)

        lidar = self._reduce_lidar(scan)
        lidar_norm = np.clip(lidar / self.LIDAR_MAX, 0.0, 1.0)

        x   = odom.pose.pose.position.x
        y   = odom.pose.pose.position.y
        qz  = odom.pose.pose.orientation.z
        qw  = odom.pose.pose.orientation.w
        yaw = 2.0 * math.atan2(qz, qw)

        dx   = self.goal_x - x
        dy   = self.goal_y - y
        dist = math.sqrt(dx * dx + dy * dy)
        angle_to_goal = math.atan2(math.sin(math.atan2(dy, dx) - yaw),
                                   math.cos(math.atan2(dy, dx) - yaw))

        return np.concatenate([
            lidar_norm,
            np.array([
                dist / 10.0,
                math.sin(angle_to_goal),
                math.cos(angle_to_goal),
                float(np.min(lidar_norm)),
            ], dtype=np.float32)
        ]).astype(np.float32)

    def _reduce_lidar(self, ranges: np.ndarray) -> np.ndarray:
        if len(ranges) < self.LIDAR_N:
            return np.pad(ranges, (0, self.LIDAR_N - len(ranges)),
                          constant_values=self.LIDAR_MAX)

        edges = np.linspace(0, len(ranges), self.LIDAR_N + 1, dtype=int)
        return np.array([
            np.min(ranges[edges[i]:edges[i + 1]])
            for i in range(self.LIDAR_N)
        ], dtype=np.float32)

    def _select_goal(self):
        if len(self._goal_positions) > 1:
            candidates = [
                position for position in self._goal_positions
                if position != (self.goal_x, self.goal_y)
            ]
            self.goal_x, self.goal_y = candidates[
                int(self._goal_rng.integers(len(candidates)))
            ]
            self._set_goal_marker(self.goal_x, self.goal_y)

    def _set_goal_marker(self, x: float, y: float):
        req = SetEntityState.Request()
        req.state.name = 'goal'
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = 0.2
        req.state.pose.orientation.w = 1.0
        req.state.reference_frame = 'world'
        self._teleport_client.call_async(req)

    def step(self, action: np.ndarray):
        self._steps_since_reset += 1

        twist = Twist()
        twist.linear.x  = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_pub.publish(twist)

        time.sleep(self.PHYSICS_WAIT)

        next_state = self.get_state()
        dist      = next_state[20] * 10.0
        min_lidar = next_state[23] * self.LIDAR_MAX

        prev_dist       = self._prev_dist
        self._prev_dist = dist

        progress = prev_dist - dist
        reward = 10.0 * progress - 0.05

        # Standing still and spinning are costly unless they lead to real progress.
        if abs(float(action[0])) < 0.03:
            reward -= 0.1
        reward -= 0.05 * abs(float(action[1]))

        if min_lidar < 0.5:
            proximity = (0.5 - min_lidar) / 0.5
            reward -= proximity * proximity

        if min_lidar < self.WALL_DETECT:
            if not self._wall_encounter_active:
                self._wall_encounter_active = True
                self._wall_encounter_dist = prev_dist
        elif self._wall_encounter_active and min_lidar > self.WALL_CLEAR:
            wall_gain = self._wall_encounter_dist - dist
            if (wall_gain > self.WALL_MIN_GAIN
                    and float(action[0]) > 0.03):
                reward += 4.0
            self._wall_encounter_active = False
            self._wall_encounter_dist = None

        done = False

        if dist < self.GOAL_THRESHOLD:
            self.node.get_logger().warn(f"GOAL REACHED  dist={dist:.3f}m")
            reward += 300.0
            done    = True
            self.cmd_pub.publish(Twist())

        elif (self._steps_since_reset > self.GRACE_STEPS
                and min_lidar < self.COLLISION_DIST):
            self.node.get_logger().warn(f"COLLISION  min={min_lidar:.3f}m")
            self.cmd_pub.publish(Twist())
            reward -= 50.0
            done    = True

        return next_state, reward, done

    def _teleport(self, x: float = 0.0, y: float = 0.0):
        self.cmd_pub.publish(Twist())  # стоп

        req = SetEntityState.Request()
        req.state.name = "burger"
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = 0.05
        req.state.pose.orientation.w = 1.0

        self._teleport_client.call_async(req)

        time.sleep(self.TELEPORT_SLEEP)

        self._prev_dist = math.sqrt(
            (self.goal_x - x)**2 + (self.goal_y - y)**2
        )
        self._wall_encounter_active = False
        self._wall_encounter_dist = None
        self.node.get_logger().info(f"TELEPORT -> ({x:.1f}, {y:.1f})")
