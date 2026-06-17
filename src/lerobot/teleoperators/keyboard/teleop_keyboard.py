#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
import os
import select
import sys
import threading
import time
import atexit
from queue import Queue
from typing import Any

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_keyboard import (
    KeyboardEndEffectorTeleopConfig,
    KeyboardRoverTeleopConfig,
    KeyboardTeleopConfig,
)

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.key_expiry = {}
        self.current_pressed = {}
        self.listener = None
        self.terminal_thread = None
        self.terminal_stop_event = None
        self.terminal_fd = None
        self.terminal_old_settings = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        pynput_connected = (
            PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()
        )
        terminal_connected = self.terminal_thread is not None and self.terminal_thread.is_alive()
        return pynput_connected or terminal_connected

    @property
    def is_calibrated(self) -> bool:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        if PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.info("pynput not available - skipping local keyboard listener.")
            self.listener = None
        self._start_terminal_listener()

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        key_char = getattr(key, "char", None)
        if key_char is not None:
            self.event_queue.put((key_char, True))
        else:
            self.event_queue.put((key, True))

    def _on_release(self, key):
        key_char = getattr(key, "char", None)
        if key_char is not None:
            self.event_queue.put((key_char, False))
        else:
            self.event_queue.put((key, False))
            if key == getattr(keyboard.Key, "ctrl", None):
                self.event_queue.put((keyboard.Key.ctrl_l, False))
                self.event_queue.put((keyboard.Key.ctrl_r, False))
            elif key == getattr(keyboard.Key, "shift", None):
                self.event_queue.put((getattr(keyboard.Key, "shift_l", keyboard.Key.shift), False))
                self.event_queue.put((keyboard.Key.shift_r, False))
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _start_terminal_listener(self) -> None:
        if termios is None or tty is None:
            return
        if not sys.stdin.isatty():
            return

        self.terminal_fd = sys.stdin.fileno()
        self.terminal_old_settings = termios.tcgetattr(self.terminal_fd)
        tty.setcbreak(self.terminal_fd)
        atexit.register(self._restore_terminal)

        self.terminal_stop_event = threading.Event()
        self.terminal_thread = threading.Thread(target=self._terminal_listener_loop, daemon=True)
        self.terminal_thread.start()
        logging.info(
            "Terminal keyboard listener enabled as fallback. Arrow keys control x/y; "
            "left/right Shift and Ctrl are handled by pynput when available; "
            "z/x and c/v remain fallback keys for z and gripper."
        )

    def _terminal_listener_loop(self) -> None:
        while self.terminal_stop_event is not None and not self.terminal_stop_event.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                continue
            char = sys.stdin.read(1)
            if char == "\x1b":
                sequence = char
                for _ in range(2):
                    readable, _, _ = select.select([sys.stdin], [], [], 0.01)
                    if readable:
                        sequence += sys.stdin.read(1)
                key = {
                    "\x1b[A": "up",
                    "\x1b[B": "down",
                    "\x1b[C": "right",
                    "\x1b[D": "left",
                }.get(sequence)
                if key is not None:
                    self.event_queue.put((key, True, time.perf_counter() + 0.2))
            elif char in {"z", "x", "c", "v"}:
                key = {
                    "z": "z_down",
                    "x": "z_up",
                    "c": "gripper_close",
                    "v": "gripper_open",
                }[char]
                self.event_queue.put((key, True, time.perf_counter() + 0.2))
            elif char in {"s", "r", "q"}:
                self.event_queue.put((char, True, time.perf_counter() + 0.2))

    def _restore_terminal(self) -> None:
        if self.terminal_fd is not None and self.terminal_old_settings is not None and termios is not None:
            termios.tcsetattr(self.terminal_fd, termios.TCSADRAIN, self.terminal_old_settings)

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            event = self.event_queue.get_nowait()
            if len(event) == 3:
                key_char, is_pressed, expires_at = event
                self.key_expiry[key_char] = expires_at
            else:
                key_char, is_pressed = event
                if not is_pressed:
                    self.key_expiry.pop(key_char, None)
            self.current_pressed[key_char] = is_pressed

        now = time.perf_counter()
        for key_char, expires_at in list(self.key_expiry.items()):
            if now >= expires_at:
                self.current_pressed[key_char] = False
                del self.key_expiry[key_char]

    def configure(self):
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()
        if self.terminal_stop_event is not None:
            self.terminal_stop_event.set()
        if self.terminal_thread is not None:
            self.terminal_thread.join(timeout=0.2)
        self._restore_terminal()
        self.terminal_thread = None
        self.terminal_stop_event = None
        self.terminal_fd = None
        self.terminal_old_settings = None


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()
        self._last_action_log_t = 0.0

    def _matches_key(self, key, pynput_names: str | tuple[str, ...], *terminal_names: str) -> bool:
        if isinstance(key, str):
            return key in terminal_names
        if not PYNPUT_AVAILABLE:
            return False
        if isinstance(pynput_names, str):
            pynput_names = (pynput_names,)
        return any(key == getattr(keyboard.Key, name, None) for name in pynput_names)

    def _pynput_key(self, name: str):
        if not PYNPUT_AVAILABLE:
            return None
        return getattr(keyboard.Key, name, None)

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1.0

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if not val:
                continue
            if self._matches_key(key, "up", "up"):
                delta_x = 1.0
            elif self._matches_key(key, "down", "down"):
                delta_x = -1.0
            elif self._matches_key(key, "left", "left"):
                delta_y = 1.0
            elif self._matches_key(key, "right", "right"):
                delta_y = -1.0
            elif self._matches_key(key, ("shift", "shift_l"), "z_down"):
                delta_z = -1.0
            elif self._matches_key(key, "shift_r", "z_up"):
                delta_z = 1.0
            elif self._matches_key(key, "ctrl_r", "gripper_open"):
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = 2.0
            elif self._matches_key(key, "ctrl_l", "gripper_close"):
                gripper_action = 0.0
            else:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        now = time.perf_counter()
        if now - self._last_action_log_t > 0.5 and (
            delta_x != 0.0 or delta_y != 0.0 or delta_z != 0.0 or gripper_action != 1.0
        ):
            logging.info(f"Keyboard EE action: {action_dict}")
            self._last_action_log_t = now

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the keyboard such as intervention status,
        episode termination, success indicators, etc.

        Keyboard mappings:
        - Any movement keys pressed = intervention active
        - 's' key = success (terminate episode successfully)
        - 'r' key = rerecord episode (terminate and rerecord)
        - 'q' key = quit episode (terminate without success)

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Check if any movement keys are currently pressed (indicates intervention)
        movement_keys = [
            self._pynput_key("up"),
            self._pynput_key("down"),
            self._pynput_key("left"),
            self._pynput_key("right"),
            self._pynput_key("shift"),
            self._pynput_key("shift_l"),
            self._pynput_key("shift_r"),
            self._pynput_key("ctrl_l"),
            self._pynput_key("ctrl_r"),
            "up",
            "down",
            "left",
            "right",
            "z_down",
            "z_up",
            "gripper_open",
            "gripper_close",
        ]
        is_intervention = any(
            self.current_pressed.get(key, False) for key in movement_keys if key is not None
        )

        # Check for episode control commands from misc_keys_queue
        terminate_episode = False
        success = False
        rerecord_episode = False

        # Process any pending misc keys
        while not self.misc_keys_queue.empty():
            key = self.misc_keys_queue.get_nowait()
            if key == "s":
                success = True
            elif key == "r":
                terminate_episode = True
                rerecord_episode = True
            elif key == "q":
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }


class KeyboardRoverTeleop(KeyboardTeleop):
    """
    Keyboard teleoperator for mobile robots like EarthRover Mini Plus.

    Provides intuitive WASD-style controls for driving a mobile robot:
    - Linear movement (forward/backward)
    - Angular movement (turning/rotation)
    - Speed adjustment
    - Emergency stop

    Keyboard Controls:
        Movement:
            - W: Move forward
            - S: Move backward
            - A: Turn left (with forward motion)
            - D: Turn right (with forward motion)
            - Q: Rotate left in place
            - E: Rotate right in place
            - X: Emergency stop

        Speed Control:
            - +/=: Increase speed
            - -: Decrease speed

        System:
            - ESC: Disconnect teleoperator

    Attributes:
        config: Teleoperator configuration
        current_linear_speed: Current linear velocity magnitude
        current_angular_speed: Current angular velocity magnitude

    Example:
        ```python
        from lerobot.teleoperators.keyboard import KeyboardRoverTeleop, KeyboardRoverTeleopConfig

        teleop = KeyboardRoverTeleop(
            KeyboardRoverTeleopConfig(linear_speed=1.0, angular_speed=1.0, speed_increment=0.1)
        )
        teleop.connect()

        while teleop.is_connected:
            action = teleop.get_action()
            robot.send_action(action)
        ```
    """

    config_class = KeyboardRoverTeleopConfig
    name = "keyboard_rover"

    def __init__(self, config: KeyboardRoverTeleopConfig):
        super().__init__(config)
        # Add rover-specific speed settings
        self.current_linear_speed = config.linear_speed
        self.current_angular_speed = config.angular_speed

    @property
    def action_features(self) -> dict:
        """Return action format for rover (linear and angular velocities)."""
        return {
            "linear_velocity": float,
            "angular_velocity": float,
        }

    @property
    def is_calibrated(self) -> bool:
        """Rover teleop doesn't require calibration."""
        return True

    def _drain_pressed_keys(self):
        """Update current_pressed state from event queue without clearing held keys"""
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                # Only remove key if it's being released
                self.current_pressed.pop(key_char, None)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Get the current action based on pressed keys.

        Returns:
            RobotAction with 'linear_velocity' and 'angular_velocity' keys.
        """
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        linear_velocity = 0.0
        angular_velocity = 0.0

        # Check which keys are currently pressed (not released)
        active_keys = {key for key, is_pressed in self.current_pressed.items() if is_pressed}

        # Linear movement (W/S) - these take priority
        if "w" in active_keys:
            linear_velocity = self.current_linear_speed
        elif "s" in active_keys:
            linear_velocity = -self.current_linear_speed

        # Turning (A/D/Q/E)
        if "d" in active_keys:
            angular_velocity = -self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "a" in active_keys:
            angular_velocity = self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "q" in active_keys:
            angular_velocity = self.current_angular_speed
            linear_velocity = 0  # Rotate in place
        elif "e" in active_keys:
            angular_velocity = -self.current_angular_speed
            linear_velocity = 0  # Rotate in place

        # Stop (X) - overrides everything
        if "x" in active_keys:
            linear_velocity = 0
            angular_velocity = 0

        # Speed adjustment
        if "+" in active_keys or "=" in active_keys:
            self.current_linear_speed += self.config.speed_increment
            self.current_angular_speed += self.config.speed_increment * self.config.angular_speed_ratio
            logging.info(
                f"Speed increased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )
        if "-" in active_keys:
            self.current_linear_speed = max(
                self.config.min_linear_speed, self.current_linear_speed - self.config.speed_increment
            )
            self.current_angular_speed = max(
                self.config.min_angular_speed,
                self.current_angular_speed - self.config.speed_increment * self.config.angular_speed_ratio,
            )
            logging.info(
                f"Speed decreased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }
