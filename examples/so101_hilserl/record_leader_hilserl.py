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

"""Record HIL-SERL demonstrations with an SO101 leader arm.

The follower is controlled with the normal LeRobot leader-to-follower joint target
path. The dataset action is recorded as HIL-SERL's 4D EE-space action:
delta_x, delta_y, delta_z, gripper.
"""

import argparse
import json
import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - POSIX-only interactive helper.
    termios = None
    tty = None

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.types import PipelineFeatureType
from lerobot.datasets.feature_utils import combine_feature_dicts
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import ObservationProcessorStep, RobotProcessorPipeline, make_default_processors
from lerobot.processor.converters import observation_to_transition, transition_to_observation
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun


MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
HIL_ACTION_FEATURES = {
    "delta_x": float,
    "delta_y": float,
    "delta_z": float,
    "gripper": float,
}
GRIPPER_LABELS = {0.0: "close", 1.0: "stay", 2.0: "open"}


class KeyReader:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd) if termios is not None and sys.stdin.isatty() else None
        if self.old_settings is not None:
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_key(self) -> str | None:
        if self.old_settings is None:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None
        return sys.stdin.read(1)


@dataclass
class RecordImageCropResizeProcessorStep(ObservationProcessorStep):
    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.resize_size is None and not self.crop_params_dict:
            return observation

        new_observation = dict(observation)
        crop_params_dict = self.crop_params_dict or {}
        for key, value in observation.items():
            if not isinstance(value, np.ndarray) or value.ndim != 3:
                continue

            image = value
            if key in crop_params_dict:
                top, left, height, width = crop_params_dict[key]
                image = image[top : top + height, left : left + width]

            if self.resize_size is not None:
                height, width = self.resize_size
                image = np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))

            new_observation[key] = image
        return new_observation

    def transform_features(self, features: dict[PipelineFeatureType, dict]) -> dict[PipelineFeatureType, dict]:
        if self.resize_size is None:
            return features
        height, width = self.resize_size
        for key, value in list(features[PipelineFeatureType.OBSERVATION].items()):
            if isinstance(value, tuple) and len(value) == 3:
                features[PipelineFeatureType.OBSERVATION][key] = (height, width, value[2])
        return features


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config-path", default=None)
    config_args, remaining_args = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--repo-id", default="ROItest/leader_hilserl_record")
    parser.add_argument("--task", default="leader arm HIL-SERL recording")
    parser.add_argument("--root", default=None)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--episode-time-s", type=float, default=60.0)
    parser.add_argument("--reset-time-s", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="hil00")
    parser.add_argument("--teleop-port", default="/dev/ttyACM1")
    parser.add_argument("--teleop-id", default="hil01")
    parser.add_argument(
        "--urdf-path",
        default="/home/ubuntu/lerobot_0.5.1/needed_files/SO-ARM100-main/Simulation/SO101/so101_new_calib.urdf",
    )
    parser.add_argument("--target-frame-name", default="gripper_frame_link")
    parser.add_argument("--top-camera", default="/dev/video0")
    parser.add_argument("--wrist-camera", default="/dev/video2")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=10)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--camera-backend", type=int, default=200)
    parser.add_argument("--x-step-size", type=float, default=0.01)
    parser.add_argument("--y-step-size", type=float, default=0.01)
    parser.add_argument("--z-step-size", type=float, default=0.01)
    parser.add_argument("--gripper-deadband", type=float, default=1.0)
    parser.add_argument("--display-data", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--streaming-encoding", action="store_true")
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument("--bounds-output-json", default="other-files/leader_record_ee_bounds.json")
    parser.add_argument("--crop-params-dict", default=None)
    parser.add_argument("--resize-size", default=None)

    if config_args.config_path is not None:
        parser.set_defaults(**load_config_defaults(config_args.config_path))

    return parser.parse_args(remaining_args)


def load_config_defaults(config_path: str) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    env = config.get("env", {})
    dataset = config.get("dataset", {})
    processor = env.get("processor", {})
    robot = env.get("robot", {})
    teleop = env.get("teleop", {})
    cameras = robot.get("cameras", {})
    top_camera = cameras.get("top", {})
    wrist_camera = cameras.get("wrist", {})
    inverse_kinematics = processor.get("inverse_kinematics", {})
    step_sizes = inverse_kinematics.get("end_effector_step_sizes", {})
    image_preprocessing = processor.get("image_preprocessing", {}) or {}

    defaults = {
        "repo_id": dataset.get("repo_id"),
        "task": dataset.get("task"),
        "root": dataset.get("root"),
        "num_episodes": dataset.get("num_episodes_to_record"),
        "fps": env.get("fps"),
        "robot_port": robot.get("port"),
        "robot_id": robot.get("id"),
        "teleop_port": teleop.get("port"),
        "teleop_id": teleop.get("id"),
        "urdf_path": inverse_kinematics.get("urdf_path"),
        "target_frame_name": inverse_kinematics.get("target_frame_name"),
        "top_camera": top_camera.get("index_or_path"),
        "wrist_camera": wrist_camera.get("index_or_path"),
        "camera_width": top_camera.get("width"),
        "camera_height": top_camera.get("height"),
        "camera_fps": top_camera.get("fps"),
        "camera_fourcc": top_camera.get("fourcc"),
        "camera_backend": top_camera.get("backend"),
        "x_step_size": step_sizes.get("x"),
        "y_step_size": step_sizes.get("y"),
        "z_step_size": step_sizes.get("z"),
        "crop_params_dict": image_preprocessing.get("crop_params_dict"),
        "resize_size": image_preprocessing.get("resize_size"),
        "push_to_hub": dataset.get("push_to_hub"),
    }

    extra = config.get("leader_hilserl_record", {})
    defaults.update(
        {
            "episode_time_s": extra.get("episode_time_s"),
            "reset_time_s": extra.get("reset_time_s"),
            "gripper_deadband": extra.get("gripper_deadband"),
            "bounds_output_json": extra.get("bounds_output_json"),
            "display_data": extra.get("display_data"),
            "vcodec": extra.get("vcodec"),
            "streaming_encoding": extra.get("streaming_encoding"),
            "encoder_threads": extra.get("encoder_threads"),
        }
    )

    return {k: v for k, v in defaults.items() if v is not None}


def parse_json_like(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def normalize_crop_params(
    crop_params_dict: dict[str, list[int] | tuple[int, int, int, int]] | None,
) -> dict[str, tuple[int, int, int, int]] | None:
    if not crop_params_dict:
        return None

    normalized = {}
    for key, value in crop_params_dict.items():
        crop = tuple(int(v) for v in value)
        if len(crop) != 4:
            raise ValueError(f"Crop params for {key} must have four values: top, left, height, width")
        normalized[key] = crop
        for prefix in ("observation.images.", "images."):
            if key.startswith(prefix):
                normalized[key.removeprefix(prefix)] = crop
    return normalized


def normalize_resize_size(resize_size: list[int] | tuple[int, int] | None) -> tuple[int, int] | None:
    if resize_size is None:
        return None
    resize = tuple(int(v) for v in resize_size)
    if len(resize) != 2:
        raise ValueError("resize_size must have two values: height, width")
    return resize


def make_record_observation_processor(args: argparse.Namespace) -> RobotProcessorPipeline:
    crop_params_dict = normalize_crop_params(parse_json_like(args.crop_params_dict))
    resize_size = normalize_resize_size(parse_json_like(args.resize_size))
    if crop_params_dict is None and resize_size is None:
        return make_default_processors()[2]

    return RobotProcessorPipeline(
        steps=[
            RecordImageCropResizeProcessorStep(
                crop_params_dict=crop_params_dict,
                resize_size=resize_size,
            )
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )


def joints_array(joints: dict[str, Any]) -> np.ndarray:
    return np.array([float(joints[f"{name}.pos"]) for name in MOTOR_NAMES], dtype=float)


def ee_xyz(kinematics: RobotKinematics, joints: dict[str, Any]) -> np.ndarray:
    return np.asarray(kinematics.forward_kinematics(joints_array(joints)), dtype=float)[:3, 3]


def gripper_action_from_diff(
    diff: float, deadband: float, positive_gripper_action: str
) -> tuple[float, str]:
    if abs(diff) <= deadband:
        return 1.0, "stay"

    if diff > 0:
        semantic = positive_gripper_action
    else:
        semantic = "open" if positive_gripper_action == "close" else "close"

    action = 0.0 if semantic == "close" else 2.0
    return action, semantic


@dataclass
class LeaderToHILActionProcessor:
    kinematics: RobotKinematics
    step_sizes: np.ndarray
    positive_gripper_action: str
    gripper_deadband: float

    def __post_init__(self):
        self.previous_gripper_target: float | None = None
        self.observed_min = np.full(3, np.inf, dtype=float)
        self.observed_max = np.full(3, -np.inf, dtype=float)
        self.target_min = np.full(3, np.inf, dtype=float)
        self.target_max = np.full(3, -np.inf, dtype=float)

    def reset(self) -> None:
        self.previous_gripper_target = None

    def _update_bounds(self, observed_xyz: np.ndarray, target_xyz: np.ndarray) -> None:
        self.observed_min = np.minimum(self.observed_min, observed_xyz)
        self.observed_max = np.maximum(self.observed_max, observed_xyz)
        self.target_min = np.minimum(self.target_min, target_xyz)
        self.target_max = np.maximum(self.target_max, target_xyz)

    def bounds(self) -> dict[str, dict[str, list[float]]]:
        return {
            "observed": {"min": self.observed_min.tolist(), "max": self.observed_max.tolist()},
            "target": {"min": self.target_min.tolist(), "max": self.target_max.tolist()},
            "combined": {
                "min": np.minimum(self.observed_min, self.target_min).tolist(),
                "max": np.maximum(self.observed_max, self.target_max).tolist(),
            },
        }

    def __call__(self, data: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, float]:
        leader_action, observation = data
        action = dict(leader_action)

        observed_xyz = ee_xyz(self.kinematics, observation)
        target_xyz = ee_xyz(self.kinematics, action)
        self._update_bounds(observed_xyz, target_xyz)

        delta = np.clip((target_xyz - observed_xyz) / self.step_sizes, -1.0, 1.0)

        gripper_target = float(action["gripper.pos"])
        if self.previous_gripper_target is None:
            gripper_diff = 0.0
        else:
            gripper_diff = gripper_target - self.previous_gripper_target
        self.previous_gripper_target = gripper_target

        gripper_action, _ = gripper_action_from_diff(
            gripper_diff, self.gripper_deadband, self.positive_gripper_action
        )

        action["delta_x"] = float(delta[0])
        action["delta_y"] = float(delta[1])
        action["delta_z"] = float(delta[2])
        action["gripper"] = float(gripper_action)
        return action


def print_gripper_status(action: float) -> None:
    print(f"\r{int(action)} {GRIPPER_LABELS[action]}      ", end="", flush=True)


def confirm_gripper_mapping(
    follower: SO101Follower,
    leader: SO101Leader,
    fps: int,
    gripper_deadband: float,
) -> str:
    positive_gripper_action = "close"
    previous_gripper_target: float | None = None

    print("请注意以下显示的状态是否正确，如果正确按 Enter 下一步，如果相反请按\"T\"转换后再次确认")
    print_gripper_status(1.0)

    with KeyReader() as keys:
        while True:
            loop_start = time.perf_counter()

            obs = follower.get_observation()
            leader_action = leader.get_action()
            follower_action = {
                f"{name}.pos": float(obs[f"{name}.pos"]) for name in MOTOR_NAMES[:-1]
            }
            follower_action["gripper.pos"] = float(leader_action["gripper.pos"])
            sent_action = follower.send_action(follower_action)

            gripper_target = float(sent_action["gripper.pos"])
            if previous_gripper_target is None:
                diff = 0.0
            else:
                diff = gripper_target - previous_gripper_target
            previous_gripper_target = gripper_target

            action, _ = gripper_action_from_diff(diff, gripper_deadband, positive_gripper_action)
            print_gripper_status(action)

            key = keys.read_key()
            if key in ("\n", "\r"):
                print()
                return positive_gripper_action
            if key is not None and key.lower() == "t":
                positive_gripper_action = "open" if positive_gripper_action == "close" else "close"
                previous_gripper_target = None
                print()
                print("请注意以下显示的状态是否正确，如果正确按 Enter 下一步，如果相反请按\"T\"转换后再次确认")

            time.sleep(max(1.0 / fps - (time.perf_counter() - loop_start), 0.0))


def create_robot(args: argparse.Namespace) -> SO101Follower:
    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=args.top_camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            fourcc=args.camera_fourcc,
            backend=args.camera_backend,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            fourcc=args.camera_fourcc,
            backend=args.camera_backend,
        ),
    }
    return SO101Follower(
        SO101FollowerConfig(
            port=args.robot_port,
            id=args.robot_id,
            cameras=cameras,
            use_degrees=True,
        )
    )


def save_bounds(path: str, bounds: dict[str, dict[str, list[float]]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bounds, indent=2), encoding="utf-8")
    print(f"Saved EE bounds to {output_path}")


def main() -> None:
    args = parse_args()
    follower = create_robot(args)
    leader = SO101Leader(SO101LeaderConfig(port=args.teleop_port, id=args.teleop_id))

    teleop_feature_processor, robot_action_processor, _ = make_default_processors()
    robot_observation_processor = make_record_observation_processor(args)
    kinematics = RobotKinematics(
        urdf_path=args.urdf_path,
        target_frame_name=args.target_frame_name,
        joint_names=MOTOR_NAMES,
    )
    step_sizes = np.array([args.x_step_size, args.y_step_size, args.z_step_size], dtype=float)
    hil_action_processor = LeaderToHILActionProcessor(
        kinematics=kinematics,
        step_sizes=step_sizes,
        positive_gripper_action="close",
        gripper_deadband=args.gripper_deadband,
    )

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=follower.observation_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=teleop_feature_processor,
            initial_features=create_initial_features(action=HIL_ACTION_FEATURES),
            use_videos=True,
        ),
    )

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.root,
        robot_type=follower.name,
        features=dataset_features,
        use_videos=True,
        image_writer_threads=4 * len(follower.cameras),
        batch_encoding_size=1,
        vcodec=args.vcodec,
        streaming_encoding=args.streaming_encoding,
        encoder_threads=args.encoder_threads,
    )

    listener = None
    try:
        leader.connect()
        follower.connect()

        hil_action_processor.positive_gripper_action = confirm_gripper_mapping(
            follower=follower,
            leader=leader,
            fps=args.fps,
            gripper_deadband=args.gripper_deadband,
        )

        if args.display_data:
            init_rerun(session_name="leader_hilserl_record")
        listener, events = init_keyboard_listener()

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < args.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}")
                hil_action_processor.reset()
                record_loop(
                    robot=follower,
                    events=events,
                    fps=args.fps,
                    teleop_action_processor=hil_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=leader,
                    dataset=dataset,
                    control_time_s=args.episode_time_s,
                    single_task=args.task,
                    display_data=args.display_data,
                )

                if not events["stop_recording"] and (
                    recorded_episodes < args.num_episodes - 1 or events["rerecord_episode"]
                ):
                    log_say("Reset the environment")
                    hil_action_processor.reset()
                    record_loop(
                        robot=follower,
                        events=events,
                        fps=args.fps,
                        teleop_action_processor=hil_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=leader,
                        control_time_s=args.reset_time_s,
                        single_task=args.task,
                        display_data=args.display_data,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", blocking=True)
        save_bounds(args.bounds_output_json, hil_action_processor.bounds())
        dataset.finalize()
        if follower.is_connected:
            follower.disconnect()
        if leader.is_connected:
            leader.disconnect()
        if listener is not None:
            listener.stop()
        if args.push_to_hub:
            dataset.push_to_hub(private=args.private)


if __name__ == "__main__":
    main()
