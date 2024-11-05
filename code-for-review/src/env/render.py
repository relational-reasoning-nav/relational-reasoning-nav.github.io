import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import patches, cm
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RenderConfig:
    robot_radius: float
    robot_fov: float
    robot_sensor_range: float
    human_radius: float
    kinematics: str = "unicycle"
    predict_steps: int = 5


def calc_fov_line_endpoint(
    ang: float, 
    point: List[float], 
    extend_factor: float
) -> List[float]:
    fov_line_rot = np.array([
        [np.cos(ang), -np.sin(ang), 0],
        [np.sin(ang), np.cos(ang), 0],
        [0, 0, 1]
    ])
    point.extend([1])
    new_point = np.matmul(fov_line_rot, np.reshape(point, [3, 1]))
    return [
        extend_factor * new_point[0, 0],
        extend_factor * new_point[1, 0],
        1
    ]


def render_scene(
    axis: plt.Axes,
    robot_state: Tuple[float, float, float, float, float],
    robot_goal: Tuple[float, float],
    human_states: List[Tuple[float, float]],
    human_visibility: List[bool],
    predicted_trajectories: Optional[np.ndarray],
    config: RenderConfig,
    save_path: str,
    idx: int,
) -> None:
    plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"
    artists = []
    
    robot_x, robot_y, robot_vx, robot_vy, robot_theta = robot_state
    goal_x, goal_y = robot_goal

    goal = mlines.Line2D(
        [goal_x], [goal_y],
        color="r",
        marker="x",
        linewidth=3,
        markersize=15,
        alpha=0.8,
    )
    axis.add_artist(goal)
    artists.append(goal)

    robot = plt.Circle(
        (robot_x, robot_y),
        config.robot_radius,
        fill=True,
        linewidth=2,
        edgecolor="r",
        facecolor=(1, 0, 0, 0.2),
    )
    axis.add_artist(robot)
    artists.append(robot)

    robot_theta = (
        robot_theta if config.kinematics == "unicycle" 
        else np.arctan2(robot_vy, robot_vx)
    )

    if config.robot_fov < 2 * np.pi:
        fov_angle = config.robot_fov / 2
        fov_line1 = mlines.Line2D([0, 0], [0, 0], linestyle="--")
        fov_line2 = mlines.Line2D([0, 0], [0, 0], linestyle="--")

        end_point_x = robot_x + config.robot_radius * np.cos(robot_theta)
        end_point_y = robot_y + config.robot_radius * np.sin(robot_theta)
        
        line_vector = [end_point_x - robot_x, end_point_y - robot_y]
        extend_factor = 20.0 / config.robot_radius

        fov_end1 = calc_fov_line_endpoint(fov_angle, line_vector.copy(), extend_factor)
        fov_line1.set_xdata(np.array([robot_x, robot_x + fov_end1[0]]))
        fov_line1.set_ydata(np.array([robot_y, robot_y + fov_end1[1]]))

        fov_end2 = calc_fov_line_endpoint(-fov_angle, line_vector.copy(), extend_factor)
        fov_line2.set_xdata(np.array([robot_x, robot_x + fov_end2[0]]))
        fov_line2.set_ydata(np.array([robot_y, robot_y + fov_end2[1]]))

        axis.add_artist(fov_line1)
        axis.add_artist(fov_line2)
        artists.extend([fov_line1, fov_line2])

    sensor_range = plt.Circle(
        (robot_x, robot_y),
        config.robot_sensor_range + config.robot_radius + config.human_radius,
        fill=False,
        linestyle="-",
        color="gray",
        alpha=0.5,
    )
    axis.add_artist(sensor_range)
    artists.append(sensor_range)

    human_circles = []
    for i, (human_x, human_y) in enumerate(human_states):
        circle = plt.Circle(
            (human_x, human_y),
            config.human_radius,
            fill=False,
            linewidth=2,
            edgecolor=cm.Set2(i // 5),
        )
        axis.add_artist(circle)
        human_circles.append(circle)
        artists.append(circle)

        if human_visibility[i]:
            circle.set_color(cm.Set2(i // 5))
        else:
            circle.set_color((0, 0, 0, 0.3))

        if predicted_trajectories is not None:
            for j in range(config.predict_steps):
                pred_circle = plt.Circle(
                    predicted_trajectories[i, (2 * j):(2 * j + 2)] + np.array([robot_x, robot_y]),
                    0.25,
                    fill=False,
                    color="tab:orange",
                    linewidth=1.5,
                    alpha=0.5,
                )
                axis.add_artist(pred_circle)
                artists.append(pred_circle)

    axis.grid(axis="both", color="black", alpha=0.1)
    plt.savefig(f"{save_path}/{idx}.jpg")

    for artist in artists:
        artist.remove()
    for text in axis.texts:
        text.set_visible(False)
