import gym
import numpy as np
from numpy.linalg import norm
import copy
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

from src.env.states import Human, Robot
from collections import namedtuple

ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
ActionRot = namedtuple('ActionRot', ['v', 'r'])


@dataclass
class CrowdSimConfig:
    arena_size: float = 6.0
    human_num: int = 20
    human_num_range: int = 0
    predict_steps: int = 5
    time_step: float = 0.25
    max_human_num: int = 25
    min_human_num: int = 15
    human_radius: float = 0.3
    robot_radius: float = 0.3
    discomfort_dist: float = 0.25
    time_limit: float = 50.0
    success_reward: float = 10.0
    collision_penalty: float = -20.0
    discomfort_penalty_factor: float = 10.0
    group_size_min: int = 5
    group_size_max: int = 5


class GroupSocialRobotNavigationEnv(gym.Env):
    def __init__(self, config: CrowdSimConfig):
        self.config = config
        self.time_step = config.time_step
        self.robot = None
        self.humans = []
        self.global_time = 0
        self.human_num = config.human_num
        self.pred_method = None
        self.case_capacity = {"train": 100000, "val": 1000, "test": 1000}
        self.case_size = {"train": 100000, "val": 1000, "test": 1000}
        self.randomize_attributes = True
        self.circle_radius = 6.0
        self.human_num_range = config.human_num_range
        
        self.max_human_num = config.max_human_num
        self.min_human_num = config.min_human_num
        self.success_reward = config.success_reward
        self.collision_penalty = config.collision_penalty
        self.discomfort_penalty_factor = config.discomfort_penalty_factor
        self.time_limit = config.time_limit
        self.discomfort_dist = config.discomfort_dist
        self.arena_size = config.arena_size
        
        self.group_size_min = config.group_size_min
        self.group_size_max = config.group_size_max
        
        self.predict_steps = config.predict_steps
        self.phase = None
        self.test_case = None
        self.case_counter = {"train": 0, "val": 0, "test": 0}
        
        self.human_visibility = None
        self.last_human_states = None
        self.human_future_traj = None
        
    def set_robot(self, robot):
        self.robot = robot
        d = {
            "robot_node": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1, 7), dtype=np.float32
            ),
            "temporal_edges": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32
            ),
            "spatial_edges": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.max_human_num, 2), dtype=np.float32
            ),
            "detected_human_num": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            "visible_masks": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.max_human_num,), dtype=bool
            ),
        }
        self.observation_space = gym.spaces.Dict(d)
        high = np.inf * np.ones([2])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def generate_random_human_position(self, human_num: int):
        group_size = np.random.randint(self.group_size_min, self.group_size_max + 1)
        num_group = (human_num - 3) // group_size
        
        for group_id in range(num_group):
            angle = None
            for _ in range(group_size):
                human, angle = self._generate_circle_crossing_human_group(angle)
                human.group_id = group_id
                self.humans.append(human)

        for _ in range(3):
            human = self._generate_circle_crossing_human_obstacle()
            human.group_id = -1
            self.humans.append(human)

    def _generate_circle_crossing_human_group(
        self, angle: Optional[float]
    ) -> Tuple[Human, float]:
        human = Human(self.config)
        if self.randomize_attributes:
            human.sample_random_attributes()

        while True:
            if angle is None:
                angle = np.random.random() * np.pi * 2
                
            noise_range = 2
            px_noise = (np.random.uniform(0, 2) - 1) * noise_range
            py_noise = (np.random.uniform(0, 2) - 1) * noise_range
            px = 9 * np.cos(angle) + px_noise
            py = 9 * np.sin(angle) + py_noise
            
            collision = False
            for agent in [self.robot] + self.humans:
                min_dist = (
                    self.circle_radius / 2
                    if self.robot.kinematics == "unicycle" and agent == self.robot
                    else human.radius + agent.radius + self.discomfort_dist
                )
                if (
                    norm((px - agent.px, py - agent.py)) < min_dist
                    or norm((px - agent.gx, py - agent.gy)) < min_dist
                ):
                    collision = True
                    break
                    
            if not collision:
                break

        human.set(px, py, -px, -py, 0, 0, 0)
        return human, angle

    def _generate_circle_crossing_human_obstacle(self) -> Human:
        human = Human(self.config)
        if self.randomize_attributes:
            human.sample_random_attributes()

        while True:
            angle = np.random.random() * np.pi * 2
            noise_range = 2
            px_noise = np.random.uniform(-2, 2) * noise_range
            py_noise = np.random.uniform(-2, 2) * noise_range
            px = px_noise
            py = py_noise
            
            collision = False
            for agent in [self.robot] + self.humans:
                min_dist = (
                    self.circle_radius / 2
                    if self.robot.kinematics == "unicycle" and agent == self.robot
                    else human.radius + agent.radius + self.discomfort_dist
                )
                if (
                    norm((px - agent.px, py - agent.py)) < min_dist
                    or norm((px - agent.gx, py - agent.gy)) < min_dist
                ):
                    collision = True
                    break
                    
            if not collision:
                break

        human.set(px, py, px, py, 0, 0, 0)
        return human

    def step(
        self, 
        action: ActionXY, 
        update: bool = True
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        if self.robot.policy.name in ["ORCA", "social_force"]:
            human_states = copy.deepcopy(self.last_human_states)
            action = self.robot.act(human_states.tolist())
        else:
            action = self.robot.policy.clip_action(action, self.robot.v_pref)

        if self.robot.kinematics == "unicycle":
            self.desired_velocity[0] = np.clip(
                self.desired_velocity[0] + action.v, -self.robot.v_pref, self.robot.v_pref
            )
            action = ActionRot(self.desired_velocity[0], action.r)

        human_actions = self.get_human_actions()

        if self.phase == "test":
            self.calc_human_future_traj(method="truth")

        reward, done, episode_info = self.calc_reward(action, danger_zone="future")

        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)

        self.global_time += self.time_step
        self.step_counter += 1
        info = {"info": episode_info}

        if self.human_num_range > 0 and self.global_time % 5 == 0:
            self._update_human_count()

        ob = self.generate_ob(reset=False, sort=self.config.args.sort_humans)
        return ob, reward, done, info


    def _update_human_count(self):
        if np.random.rand() < 0.5:
            if len(self.observed_human_ids) == 0:
                max_remove_num = self.human_num - self.min_human_num
            else:
                max_remove_num = min(
                    self.human_num - self.min_human_num,
                    (self.human_num - 1) - max(self.observed_human_ids),
                )
            remove_num = np.random.randint(low=0, high=max_remove_num + 1)
            for _ in range(remove_num):
                self.humans.pop()
            self.human_num -= remove_num
            self.last_human_states = self.last_human_states[: self.human_num]
        else:
            add_num = np.random.randint(low=0, high=self.human_num_range + 1)
            if add_num > 0:
                true_add_num = 0
                for i in range(self.human_num, self.human_num + add_num):
                    if i == self.config.sim.human_num + self.human_num_range:
                        break
                    self.generate_random_human_position(human_num=1)
                    self.humans[i].id = i
                    true_add_num += 1
                self.human_num += true_add_num
                if true_add_num > 0:
                    self.last_human_states = np.concatenate(
                        (
                            self.last_human_states,
                            np.array([[15, 15, 0, 0, 0.3]] * true_add_num),
                        ),
                        axis=0,
                    )


    def calc_reward(
        self, action: ActionXY, danger_zone: str = "circle"
    ) -> Tuple[float, bool, Any]:
        dmin = float("inf")
        danger_dists = []
        collision = False

        for human in self.humans:
            dx = human.px - self.robot.px
            dy = human.py - self.robot.py
            closest_dist = (dx**2 + dy**2) ** (1 / 2) - human.radius - self.robot.radius

            if closest_dist < self.discomfort_dist:
                danger_dists.append(closest_dist)
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        goal_radius = 0.6 if self.robot.kinematics == "unicycle" else self.robot.radius
        reaching_goal = norm(
            np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position())
        ) < goal_radius

        if danger_zone == "circle" or self.phase == "train":
            danger_cond = dmin < self.discomfort_dist
            min_danger_dist = 0
        else:
            relative_pos = self.human_future_traj[1:, :, :2] - np.array(
                [self.robot.px, self.robot.py]
            )
            relative_dist = np.linalg.norm(relative_pos, axis=-1)
            collision_idx = relative_dist < self.robot.radius + self.config.humans.radius
            danger_cond = np.any(collision_idx)
            min_danger_dist = (
                np.amin(relative_dist[collision_idx]) if danger_cond else 0
            )

        if self.global_time >= self.time_limit - 1:
            return 0, True, "Timeout"
        elif collision:
            return self.collision_penalty, True, "Collision"
        elif reaching_goal:
            return self.success_reward, True, "Success"
        elif danger_cond:
            reward = (
                (dmin - self.discomfort_dist)
                * self.discomfort_penalty_factor
                * self.time_step
            )
            return reward, False, "CollisionGroup"
        else:
            pot_factor = 2 if self.robot.kinematics == "holonomic" else 3
            potential_cur = np.linalg.norm(
                np.array([self.robot.px, self.robot.py])
                - np.array(self.robot.get_goal_position())
            )
            reward = pot_factor * (-abs(potential_cur) - self.potential)
            self.potential = -abs(potential_cur)

            if self.robot.kinematics == "unicycle":
                reward = reward - 4.5 * action.r**2 - (2 * abs(action.v) if action.v < 0 else 0)

            return reward, False, None


    def generate_ob(
        self, reset: bool = False, sort: bool = False
    ) -> Dict[str, np.ndarray]:
        visible_humans, num_visibles, self.human_visibility = self.get_num_human_in_fov()
        ob = {
            "robot_node": self.robot.get_full_state_list_noV(),
            "temporal_edges": np.array([self.robot.vx, self.robot.vy]),
        }

        prev_human_pos = copy.deepcopy(self.last_human_states)
        self.update_last_human_states(self.human_visibility, reset=reset)

        all_spatial_edges = np.ones((self.max_human_num, 2)) * np.inf
        for i in range(self.human_num):
            if self.human_visibility[i]:
                relative_pos = np.array(
                    [
                        self.last_human_states[i, 0] - self.robot.px,
                        self.last_human_states[i, 1] - self.robot.py,
                    ]
                )
                all_spatial_edges[self.humans[i].id, :2] = relative_pos

        ob["visible_masks"] = np.zeros(self.max_human_num, dtype=bool)
        if sort:
            ob["spatial_edges"] = np.array(
                sorted(all_spatial_edges, key=lambda x: np.linalg.norm(x))
            )
            if num_visibles > 0:
                ob["visible_masks"][:num_visibles] = True
        else:
            ob["spatial_edges"] = all_spatial_edges
            ob["visible_masks"][: self.human_num] = self.human_visibility

        ob["spatial_edges"][np.isinf(ob["spatial_edges"])] = 15
        ob["detected_human_num"] = max(num_visibles, 1)
        self.observed_human_ids = np.where(self.human_visibility)[0]
        self.ob = ob

        return ob
