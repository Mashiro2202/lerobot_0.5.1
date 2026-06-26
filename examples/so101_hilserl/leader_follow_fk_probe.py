#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Follow an SO101 leader arm while probing EE-space values for HIL-SERL.

This script keeps the normal LeRobot leader-to-follower teleoperation path intact:
leader joints are read, processed by the default processors, and sent to the follower.
In parallel, it computes FK-based XYZ values and writes a CSV for checking bounds and
the 4D HIL-SERL action representation.

Use --gripper-only to lock the first five joints at the current follower position and
only pass the leader gripper target through. That mode is intended for manually
checking whether positive gripper motion should mean close or open.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig


MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="hil00")
    parser.add_argument("--teleop-port", default="/dev/ttyACM1")
    parser.add_argument("--teleop-id", default="hil01")
    parser.add_argument(
        "--urdf-path",
        default="/home/ubuntu/lerobot_0.5.1/needed_files/SO-ARM100-main/Simulation/SO101/so101_new_calib.urdf",
    )
    parser.add_argument("--target-frame-name", default="gripper_frame_link")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--output-csv", default="leader_follow_fk_probe.csv")
    parser.add_argument("--report-interval-s", type=float, default=1.0)
    parser.add_argument("--x-step-size", type=float, default=0.01)
    parser.add_argument("--y-step-size", type=float, default=0.01)
    parser.add_argument("--z-step-size", type=float, default=0.01)
    parser.add_argument("--gripper-deadband", type=float, default=1.0)
    parser.add_argument(
        "--positive-gripper-action",
        choices=["close", "open"],
        default="close",
        help="Semantic label for target gripper increasing. close -> 0, open -> 2.",
    )
    parser.add_argument(
        "--gripper-only",
        action="store_true",
        help="Lock the first five follower joints and only pass through the leader gripper target.",
    )
    return parser.parse_args()


def joints_array(joints: dict[str, float]) -> np.ndarray:
    return np.array([float(joints[f"{name}.pos"]) for name in MOTOR_NAMES], dtype=float)


def ee_xyz(kinematics: RobotKinematics, joints: dict[str, float]) -> np.ndarray:
    return np.asarray(kinematics.forward_kinematics(joints_array(joints)), dtype=float)[:3, 3]


def update_bounds(bounds: dict[str, np.ndarray], xyz: np.ndarray) -> None:
    bounds["min"] = np.minimum(bounds["min"], xyz)
    bounds["max"] = np.maximum(bounds["max"], xyz)


def gripper_action_from_diff(
    diff: float, deadband: float, positive_gripper_action: str
) -> tuple[float, str]:
    if abs(diff) <= deadband:
        return 1.0, "stay"

    if diff > 0:
        semantic = positive_gripper_action
    else:
        semantic = "open" if positive_gripper_action == "close" else "close"

    return (0.0, "close") if semantic == "close" else (2.0, "open")


def print_bounds(label: str, bounds: dict[str, np.ndarray]) -> None:
    print(
        f"{label}: min={np.round(bounds['min'], 4).tolist()} "
        f"max={np.round(bounds['max'], 4).tolist()}"
    )


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    follower = SO101Follower(SO101FollowerConfig(port=args.robot_port, id=args.robot_id))
    leader = SO101Leader(SO101LeaderConfig(port=args.teleop_port, id=args.teleop_id))
    teleop_action_processor, robot_action_processor, _ = make_default_processors()

    kinematics = RobotKinematics(
        urdf_path=args.urdf_path,
        target_frame_name=args.target_frame_name,
        joint_names=MOTOR_NAMES,
    )

    observed_bounds = {
        "min": np.full(3, np.inf, dtype=float),
        "max": np.full(3, -np.inf, dtype=float),
    }
    target_bounds = {
        "min": np.full(3, np.inf, dtype=float),
        "max": np.full(3, -np.inf, dtype=float),
    }

    fieldnames = [
        "elapsed_s",
        "observed_x",
        "observed_y",
        "observed_z",
        "target_x",
        "target_y",
        "target_z",
        "delta_x",
        "delta_y",
        "delta_z",
        "gripper_action",
        "gripper_semantic",
        "gripper_target_diff",
        *[f"leader_{name}" for name in MOTOR_NAMES],
        *[f"follower_observed_{name}" for name in MOTOR_NAMES],
        *[f"follower_target_{name}" for name in MOTOR_NAMES],
    ]

    print("Connecting leader and follower...")
    leader.connect()
    follower.connect()

    previous_gripper_target = None
    start = time.perf_counter()
    last_report = start

    try:
        with output_csv.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            print("Starting official leader -> follower follow loop with FK probe.")
            if args.gripper_only:
                print("GRIPPER-ONLY MODE: first five follower joints are locked.")
            print(f"Writing CSV to {output_csv}")

            while True:
                loop_start = time.perf_counter()
                elapsed_s = loop_start - start
                if args.duration_s is not None and elapsed_s >= args.duration_s:
                    break

                obs = follower.get_observation()
                raw_action = leader.get_action()
                teleop_action = teleop_action_processor((raw_action, obs))
                robot_action_to_send = robot_action_processor((teleop_action, obs))

                if args.gripper_only:
                    robot_action_to_send = {
                        f"{name}.pos": float(obs[f"{name}.pos"]) for name in MOTOR_NAMES[:-1]
                    } | {"gripper.pos": float(robot_action_to_send["gripper.pos"])}

                sent_action = follower.send_action(robot_action_to_send)

                observed_xyz = ee_xyz(kinematics, obs)
                target_xyz = ee_xyz(kinematics, sent_action)
                update_bounds(observed_bounds, observed_xyz)
                update_bounds(target_bounds, target_xyz)

                step_sizes = np.array(
                    [args.x_step_size, args.y_step_size, args.z_step_size], dtype=float
                )
                delta = np.clip((target_xyz - observed_xyz) / step_sizes, -1.0, 1.0)

                gripper_target = float(sent_action["gripper.pos"])
                if previous_gripper_target is None:
                    gripper_diff = 0.0
                else:
                    gripper_diff = gripper_target - previous_gripper_target
                previous_gripper_target = gripper_target

                gripper_action, gripper_semantic = gripper_action_from_diff(
                    gripper_diff, args.gripper_deadband, args.positive_gripper_action
                )

                row = {
                    "elapsed_s": round(elapsed_s, 4),
                    "observed_x": observed_xyz[0],
                    "observed_y": observed_xyz[1],
                    "observed_z": observed_xyz[2],
                    "target_x": target_xyz[0],
                    "target_y": target_xyz[1],
                    "target_z": target_xyz[2],
                    "delta_x": delta[0],
                    "delta_y": delta[1],
                    "delta_z": delta[2],
                    "gripper_action": gripper_action,
                    "gripper_semantic": gripper_semantic,
                    "gripper_target_diff": gripper_diff,
                }
                for name in MOTOR_NAMES:
                    row[f"leader_{name}"] = raw_action[f"{name}.pos"]
                    row[f"follower_observed_{name}"] = obs[f"{name}.pos"]
                    row[f"follower_target_{name}"] = sent_action[f"{name}.pos"]
                writer.writerow(row)

                now = time.perf_counter()
                if now - last_report >= args.report_interval_s:
                    print(
                        f"t={elapsed_s:.1f}s "
                        f"xyz={np.round(observed_xyz, 4).tolist()} "
                        f"delta={np.round(delta, 3).tolist()} "
                        f"gripper={gripper_action:.0f}/{gripper_semantic} "
                        f"diff={gripper_diff:.2f}"
                    )
                    print_bounds("observed_bounds", observed_bounds)
                    print_bounds("target_bounds", target_bounds)
                    if args.gripper_only:
                        print(
                            "Check the real gripper: if positive diff closes it, keep "
                            "--positive-gripper-action close; otherwise use open."
                        )
                    last_report = now

                precise_sleep_s = max(1.0 / args.fps - (time.perf_counter() - loop_start), 0.0)
                time.sleep(precise_sleep_s)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nFinal EE bounds:")
        print(json.dumps({"observed": {k: v.tolist() for k, v in observed_bounds.items()}}, indent=2))
        print(json.dumps({"target": {k: v.tolist() for k, v in target_bounds.items()}}, indent=2))
        leader.disconnect()
        follower.disconnect()


if __name__ == "__main__":
    main()
