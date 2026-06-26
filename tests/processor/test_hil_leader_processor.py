import numpy as np
import torch

from lerobot.processor.hil_processor import (
    TELEOP_ACTION_KEY,
    LeaderJointsToHILActionProcessorStep,
)
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.types import TransitionKey


MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class FakeKinematics:
    def forward_kinematics(self, q):
        transform = np.eye(4)
        transform[:3, 3] = q[:3]
        return transform


def joints(x, y, z, wrist_flex=0.0, wrist_roll=0.0, gripper=40.0):
    return {
        "shoulder_pan.pos": x,
        "shoulder_lift.pos": y,
        "elbow_flex.pos": z,
        "wrist_flex.pos": wrist_flex,
        "wrist_roll.pos": wrist_roll,
        "gripper.pos": gripper,
    }


def make_processor(**kwargs):
    return LeaderJointsToHILActionProcessorStep(
        follower_kinematics=FakeKinematics(),
        leader_motor_names=MOTOR_NAMES,
        follower_motor_names=MOTOR_NAMES,
        end_effector_step_sizes={"x": 0.01, "y": 0.02, "z": 0.03},
        **kwargs,
    )


def make_transition(follower, leader):
    return {
        TransitionKey.ACTION: torch.tensor([0.0, 0.0, 0.0, 1.0]),
        TransitionKey.OBSERVATION: follower,
        TransitionKey.COMPLEMENTARY_DATA: {TELEOP_ACTION_KEY: leader},
        TransitionKey.INFO: {},
    }


def test_first_leader_frame_latches_reference_and_keeps_follower_still():
    processor = make_processor()
    transition = make_transition(
        follower=joints(0.10, -0.02, 0.01, gripper=40.0),
        leader=joints(0.50, 0.50, 0.50, gripper=60.0),
    )

    output = processor(transition)

    assert output[TransitionKey.ACTION] == joints(0.10, -0.02, 0.01, gripper=40.0)
    assert output[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY].tolist() == [
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert output[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION] is False


def test_leader_joint_delta_is_mirrored_to_follower_and_recorded_as_4d_action():
    processor = make_processor(gripper_deadband=1.0)
    processor(
        make_transition(
            follower=joints(0.10, -0.02, 0.01, gripper=40.0),
            leader=joints(0.50, 0.50, 0.50, gripper=60.0),
        )
    )

    output = processor(
        make_transition(
            follower=joints(0.10, -0.02, 0.01, gripper=40.0),
            leader=joints(0.505, 0.48, 0.515, gripper=70.0),
        )
    )
    follower_action = output[TransitionKey.ACTION]
    recorded_action = output[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]

    assert np.isclose(follower_action["shoulder_pan.pos"], 0.105)
    assert np.isclose(follower_action["shoulder_lift.pos"], -0.04)
    assert np.isclose(follower_action["elbow_flex.pos"], 0.025)
    assert np.isclose(follower_action["gripper.pos"], 50.0)
    assert recorded_action.shape == (4,)
    assert np.isclose(recorded_action[0].item(), 0.5)
    assert np.isclose(recorded_action[1].item(), -1.0)
    assert np.isclose(recorded_action[2].item(), 0.5)
    assert recorded_action[3].item() == 0.0
    assert output[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION] is True


def test_leader_gripper_recording_uses_target_vs_current_follower_position():
    close_processor = make_processor(gripper_deadband=1.0)
    close_processor(
        make_transition(
            follower=joints(0.0, 0.0, 0.0, gripper=40.0),
            leader=joints(0.0, 0.0, 0.0, gripper=40.0),
        )
    )
    close_action = close_processor(
        make_transition(
            follower=joints(0.0, 0.0, 0.0, gripper=40.0),
            leader=joints(0.0, 0.0, 0.0, gripper=45.0),
        )
    )[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]
    stay_action = close_processor(
        make_transition(
            follower=joints(0.0, 0.0, 0.0, gripper=40.0),
            leader=joints(0.0, 0.0, 0.0, gripper=45.0),
        )
    )[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]

    open_processor = make_processor(gripper_deadband=1.0)
    open_processor(
        make_transition(
            follower=joints(0.0, 0.0, 0.0, gripper=40.0),
            leader=joints(0.0, 0.0, 0.0, gripper=40.0),
        )
    )
    open_action = open_processor(
        make_transition(
            follower=joints(0.0, 0.0, 0.0, gripper=40.0),
            leader=joints(0.0, 0.0, 0.0, gripper=35.0),
        )
    )[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]

    assert close_action[3].item() == 0.0
    assert open_action[3].item() == 2.0
    assert stay_action[3].item() == 1.0
