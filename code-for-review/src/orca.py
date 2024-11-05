import numpy as np
import rvo2
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ORCAParams:
    neighbor_dist: float = 10.0
    safety_space: float = 0.15
    time_horizon: float = 5.0
    time_horizon_obst: float = 5.0
    max_speed: float = 1.0
    time_step: float = 0.25


class ActionXY:
    def __init__(self, vx: float, vy: float):
        self.vx = vx
        self.vy = vy


class ORCA:
    def __init__(self, params: Optional[ORCAParams] = None):
        self.params = params or ORCAParams()
        self.max_neighbors = None
        self.radius = None
        self.max_speed = self.params.max_speed
        self.sim: Optional[rvo2.PyRVOSimulator] = None
        self.safety_space = self.params.safety_space
        self.time_step = self.params.time_step
        self.last_state = None

    @staticmethod
    def reach_destination(state) -> bool:
        self_state = state.self_state
        if np.linalg.norm(
            (self_state.py - self_state.gy, self_state.px - self_state.gx)
        ) < self_state.radius:
            return True
        return False

    def predict(self, state) -> ActionXY:
        self_state = state.self_state
        self.max_neighbors = len(state.human_states)
        self.radius = state.self_state.radius
        
        params = (
            self.params.neighbor_dist,
            self.max_neighbors,
            self.params.time_horizon,
            self.params.time_horizon_obst,
        )

        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None

        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(
                self.time_step,
                *params,
                self.radius,
                self.max_speed,
            )
            
            self.sim.addAgent(
                (self_state.px, self_state.py),
                *params,
                self_state.radius + 0.01 + self.safety_space,
                self_state.v_pref,
                (self_state.vx, self_state.vy),
            )

            for human_state in state.human_states:
                self.sim.addAgent(
                    (human_state.px, human_state.py),
                    *params,
                    human_state.radius + 0.01 + self.safety_space,
                    self.max_speed,
                    (human_state.vx, human_state.vy),
                )
        else:
            self.sim.setAgentPosition(0, (self_state.px, self_state.py))
            self.sim.setAgentVelocity(0, (self_state.vx, self_state.vy))
            
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, (human_state.px, human_state.py))
                self.sim.setAgentVelocity(i + 1, (human_state.vx, human_state.vy))

        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i in range(len(state.human_states)):
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))
        self.last_state = state

        return action