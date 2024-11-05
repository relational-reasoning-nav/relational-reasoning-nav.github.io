import numpy as np
from numpy.linalg import norm
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

from policy.policy_factory import policy_factory
from envs.utils.action import ActionXY, ActionRot
from envs.utils.state import ObservableState, FullState, JointState


@dataclass
class AgentConfig:
    visible: bool = True
    v_pref: float = 1.0
    radius: float = 0.3
    sensor: str = "coordinates"
    fov: float = 2.0
    kinematics: str = "holonomic"
    time_step: float = 0.25
    sensor_range: float = 5.0
    policy_name: str = "none"


class Human:
    def __init__(self, config: AgentConfig):
        self.visible = config.visible
        self.v_pref = config.v_pref
        self.radius = config.radius
        self.policy = policy_factory[config.policy_name](config)
        self.sensor = config.sensor
        self.fov = np.pi * config.fov
        self.kinematics = "holonomic"
        self.time_step = config.time_step
        self.policy.time_step = config.time_step
        
        self.px: Optional[float] = None
        self.py: Optional[float] = None
        self.gx: Optional[float] = None
        self.gy: Optional[float] = None
        self.vx: Optional[float] = None
        self.vy: Optional[float] = None
        self.theta: Optional[float] = None
        
        self.is_obstacle = False
        self.id = None
        self.observed_id = -1
        self.group_id = 0

    def sample_random_attributes(self):
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(
        self,
        px: float,
        py: float,
        gx: float,
        gy: float,
        vx: float,
        vy: float,
        theta: float,
        radius: Optional[float] = None,
        v_pref: Optional[float] = None,
    ):
        self.px, self.py = px, py
        self.gx, self.gy = gx, gy
        self.vx, self.vy = vx, vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_observable_state(self) -> ObservableState:
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_full_state(self) -> FullState:
        return FullState(
            self.px, self.py, self.vx, self.vy, 
            self.radius, self.gx, self.gy, self.v_pref, self.theta
        )

    def get_position(self) -> Tuple[float, float]:
        return self.px, self.py

    def get_goal_position(self) -> Tuple[float, float]:
        return self.gx, self.gy

    def act(self, ob: List[ObservableState]) -> Any:
        state = JointState(self.get_full_state(), ob)
        return self.policy.predict(state)

    def act_joint_state(self, ob: JointState) -> Any:
        return self.policy.predict(ob)

    def check_validity(self, action: Any):
        if self.kinematics == "holonomic":
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action: Any, delta_t: float) -> Tuple[float, float]:
        self.check_validity(action)
        if self.kinematics == "holonomic":
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
            return px, py

        epsilon = 0.0001
        if abs(action.r) < epsilon:
            R = 0
        else:
            w = action.r / delta_t
            R = action.v / w

        px = self.px - R * np.sin(self.theta) + R * np.sin(self.theta + action.r)
        py = self.py + R * np.cos(self.theta) - R * np.cos(self.theta + action.r)
        return px, py

    def step(self, action: Any):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        
        if self.kinematics == "holonomic":
            self.vx = action.vx
            self.vy = action.vy
        else:
            self.theta = (self.theta + action.r) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)

    def reached_destination(self) -> bool:
        return norm(
            np.array(self.get_position()) - np.array(self.get_goal_position())
        ) < self.radius


@dataclass
class RobotState:
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    radius: float
    goal: Tuple[float, float]
    v_pref: float
    heading: float


class Robot(Human):
    def build_state(self) -> RobotState:
        return RobotState(
            position=self.get_position(),
            velocity=self.get_velocity(),
            radius=self.radius,
            goal=self.get_goal_position(),
            v_pref=self.v_pref,
            heading=self.theta,
        )

    def is_goal_reached(self) -> bool:
        return super().reached_destination()

    def compute_next_state(self, action: Any) -> RobotState:
        self.check_validity(action)
        next_px, next_py = self.compute_position(action, self.time_step)
        
        if self.kinematics == "holonomic":
            next_vx, next_vy = action.vx, action.vy
            next_theta = self.theta
        else:
            next_theta = (self.theta + action.r) % (2 * np.pi)
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
            
        return RobotState(
            position=(next_px, next_py),
            velocity=(next_vx, next_vy),
            radius=self.radius,
            goal=self.get_goal_position(),
            v_pref=self.v_pref,
            heading=next_theta,
        )

    def set_policy(self, policy: Any):
        if hasattr(policy, "set_phase"):
            policy.set_phase({"robot": self})
        self.policy = policy

    def update_state(self, state: RobotState):
        self.px, self.py = state.position
        self.vx, self.vy = state.velocity
        self.theta = state.heading
        self.gx, self.gy = state.goal
        self.v_pref = state.v_pref
        self.radius = state.radius

    def sense(self, humans: List[Human]) -> List[ObservableState]:
        observed_states = []
        for human in humans:
            dx = human.px - self.px
            dy = human.py - self.py
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist > self.sensor_range:
                continue

            if self.fov < 2 * np.pi:
                angle = np.arctan2(dy, dx)
                half_fov = self.fov / 2
                if (
                    abs(angle - self.theta) > half_fov
                    and abs(angle - self.theta) < 2 * np.pi - half_fov
                ):
                    continue

            observed_states.append(human.get_observable_state())
            
        return observed_states