# SO101 Leader HIL-SERL Recording

This example records HIL-SERL demonstrations with an SO101 leader arm while keeping the follower
control path close to the standard LeRobot leader-to-follower teleoperation behavior.

The follower receives 6D joint targets from the leader arm. The dataset stores the HIL-SERL
end-effector action format:

```text
action = [delta_x, delta_y, delta_z, gripper]
```

where `gripper` uses:

```text
0 = close
1 = stay
2 = open
```

With the reference config, `observation.state` contains 18 values:

```text
6 joint positions + 6 joint velocities + 6 motor currents
```

## Run

Pull the branch and run the config-driven recorder:

```bash
cd /home/ubuntu/lerobot_0.5.1
git pull --ff-only origin leader-hilserl-ee-delta
python examples/so101_hilserl/record_leader_hilserl.py \
  --config-path examples/so101_hilserl/record_leader_hilserl_config.json
```

At startup, the script checks the gripper direction:

```text
Check whether the displayed gripper state matches the real robot. Press Enter to continue if it is correct, or press "T" to invert it and check again.
1 stay
```

Press `Enter` when the displayed state matches the real robot. Press `T` if close/open is inverted.

During recording, the standard LeRobot keyboard controls are available:

```text
Right arrow = finish the current episode/reset segment
Left arrow  = rerecord the current episode
Esc         = stop recording
```

## Configuration

Use `record_leader_hilserl_config.json` as the clean reference configuration. It follows the same
top-level structure used by the HIL-SERL environment configs:

```text
env
dataset
leader_hilserl_record
mode
device
```

The recorder reads these fields from the config:

```text
env.fps
env.robot
env.teleop
env.processor.image_preprocessing
env.processor.inverse_kinematics
dataset.repo_id
dataset.root
dataset.task
dataset.num_episodes_to_record
dataset.push_to_hub
leader_hilserl_record
```

`image_preprocessing.crop_params_dict` and `image_preprocessing.resize_size` are applied during
recording, so the saved dataset can directly contain cropped `128x128` videos.

## Check The Dataset

Inspect the recorded dataset:

```bash
python -m lerobot.scripts.lerobot_info --repo-id ROItest/leader_hilserl_record_config_test
```

The expected action feature is:

```text
action {'dtype': 'float32', 'shape': (4,), 'names': {'delta_x': 0, 'delta_y': 1, 'delta_z': 2, 'gripper': 3}}
observation.state {'dtype': 'float32', 'shape': (18,), ...}
```

Check the gripper action distribution:

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

ds = LeRobotDataset("ROItest/leader_hilserl_record_config_test")
a = torch.stack([ds[i]["action"] for i in range(len(ds))])
s = torch.stack([ds[i]["observation.state"] for i in range(len(ds))])
print("action shape:", a.shape)
print("state shape:", s.shape)
print("min:", a.min(dim=0).values.tolist())
print("max:", a.max(dim=0).values.tolist())
print("gripper values:", sorted(set(a[:, 3].tolist())))
print("saturated xyz counts:", ((a[:, :3].abs() >= 0.999).sum(dim=0)).tolist())
PY
```

The recorder also writes end-effector bounds to:

```text
other-files/leader_record_ee_bounds.json
```
