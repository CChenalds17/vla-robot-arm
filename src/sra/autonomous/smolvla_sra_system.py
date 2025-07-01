from sra.servo_controller import ServoController
from sra.camera_handler import CameraHandler
from sra.helpers import convert_angle_to_arduino, convert_angle_to_vla, tensor_to_pil

from sra.aria_common import ctrl_c_handler

import torch
import cv2
import threading
import queue
import time

from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Servo constants
INIT_ANGLE = 0

class SmolVLASRASystem:
    """
    System integrating Aria glasses, SmolVLA, and Arduino. Treats angle range as [-180, +180] and **converts every angle before transmission with Arduino**.
    """
    def __init__(self, arduino_port, baud_rate, streaming_interface, update_iptables, profile_name, device_ip, control_hz=10.0, \
                 print_outputs=False, dataset_path="cchenalds17/svla_custom_aria_1dof", model_path="lerobot/smolvla_base", device="mps"):
        # Initialize components (parameters supplied by command-line arguments)
        self.print_outputs = print_outputs

        # Set up gating for prediction frequency
        self.control_interval = 1.0 / control_hz   # Seconds between predictions
        self._last_control_time = 0.0

        self.device = device
        self.dataset = LeRobotDataset(dataset_path)
        self.policy = SmolVLAPolicy.from_pretrained(model_path)
        self.setup_pol_state_dict()

        self.servo_controller = ServoController(arduino_port, baud_rate, convert_angle_to_arduino(INIT_ANGLE), print_outputs)
        self.camera_handler = CameraHandler(streaming_interface, update_iptables, profile_name, device_ip)

        self.current_frame = None

        # Task queue and current task
        self.task_queue = queue.Queue()
        self.current_task = ""
        # Prompt thread (to ensure only 1 runs at a time)
        self._prompting = threading.Event()

        # Display windows
        self.undistorted_window = "Undistorted Feed"
    
    def setup_pol_state_dict(self):
        """Set up policy state dictionary with dataset's mean and std"""
        pol_state_dict = self.policy.state_dict()

        pol_state_dict['normalize_inputs.buffer_observation_state.mean'] = torch.from_numpy(
            self.dataset.meta.stats['observation.state']['mean'])
        pol_state_dict['normalize_inputs.buffer_observation_state.std'] = torch.from_numpy(
            self.dataset.meta.stats['observation.state']['std'])
        pol_state_dict['normalize_targets.buffer_action.mean'] = torch.from_numpy(self.dataset.meta.stats['action']['mean'])
        pol_state_dict['normalize_targets.buffer_action.std'] = torch.from_numpy(self.dataset.meta.stats['action']['std'])
        pol_state_dict['unnormalize_outputs.buffer_action.mean'] = torch.from_numpy(self.dataset.meta.stats['action']['mean'])
        pol_state_dict['unnormalize_outputs.buffer_action.std'] = torch.from_numpy(self.dataset.meta.stats['action']['std'])

        self.policy.load_state_dict(pol_state_dict)
    
    def _prompt_for_task(self):
        """Blocking prompt run in background thread. Does not interrupt main video/control loop"""
        try:
            self.current_task = ""
            new_task = input("\nEnter new task: ")
            self.task_queue.put(new_task.strip())
            print(f"Queued new task: {new_task}")
        except EOFError:
            print("Prompt cancelled (EOF)")
        finally:
            self._prompting.clear()

    def run_control_loop(self):
        """Control loop to operate SRA"""
        cv2.namedWindow(self.undistorted_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.undistorted_window, 512, 512)
        cv2.setWindowProperty(self.undistorted_window, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(self.undistorted_window, 1200, 50)

        with ctrl_c_handler() as ctrl_c:
            while not (((key := (cv2.waitKey(1) & 0xFF)) in (27, ord('q'))) or ctrl_c):
                # Get frame from camera handler
                undistorted_display = self.camera_handler.get_processed_frame()
                # Update current_frame if there is a new frame
                if undistorted_display is not None:
                    self.current_frame = undistorted_display

                # If we haven't gotten our initial frame yet, block until we have it
                if self.current_frame is None:
                    key = cv2.waitKey(1) & 0xFF
                    continue
                # Show current frame
                cv2.imshow(self.undistorted_window, self.current_frame)

                # Spawn background prompt on 't'
                if key == ord('t'):
                    # Check that prompt thread isn't already running
                    if not self._prompting.is_set():
                        self._prompting.set()
                        threading.Thread(target=self._prompt_for_task, daemon=True).start()

                # Check for any new tasks and update
                try:
                    new_task = self.task_queue.get_nowait()
                    self.current_task = new_task
                    print(f"Switched to new task: {self.current_task}")
                except queue.Empty:
                    pass

                # Automatic control step (according to desired frequency)
                now = time.perf_counter()
                if now - self._last_control_time >= self.control_interval:
                    self._last_control_time = now
                    
                    # If there is no task yet, prompt user for task
                    if self.current_task == "":
                        if not self._prompting.is_set():
                            self._prompting.set()
                            print(f"\nYou must enter a task first")
                            threading.Thread(target=self._prompt_for_task, daemon=True).start()
                        continue
                    
                    # If there is a task, get the prediction
                    observation = self.get_observation()
                    action = self.policy.select_action(observation)

                    if self.print_outputs:
                        print(f"VLA Action: {action}")
                    servo_angle = SmolVLASRASystem.action_to_angle(action)
                    self.servo_controller.move_servo(convert_angle_to_arduino(servo_angle))
        
        self.camera_handler.terminate()

    def get_observation(self):
        """Fetches observation data and formats it for SmolVLA"""
        obs = convert_angle_to_vla(self.servo_controller.get_status()) # Gets servo angle from Arduino and converts it
        observation_state = SmolVLASRASystem.obs_to_smolvla_state(obs).to(self.device)
        observation_image = SmolVLASRASystem.img_to_smolvla_tensor(self.current_frame).to(self.device)
        task = self.current_task

        observation = {
            "observation.state": observation_state,
            "observation.image": observation_image,
            "task": [task]
        }

        if self.print_outputs:
            print('---------------------------------------------------------')
            print(f"Observation State: {observation_state}")
            tensor_to_pil(observation_image).show()
            print(f"Observation Image Top: {observation_image}")
            print(f"Observation Image Top Shape: {observation_image.shape}")
            print(f"Task: {task}")
            print('---------------------------------------------------------')

        return observation
        
    @staticmethod
    def img_to_smolvla_tensor(img):
        """Returns undistorted_tensor: a Tensor of shape [1, 3, 322, 322] processed for SmolVLA input"""
        undistorted_tensor = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=torch.float32) # Convert to RGB Tensor
        undistorted_tensor = undistorted_tensor.permute(2, 0, 1).unsqueeze(0) # Process dimensions
        undistorted_tensor = undistorted_tensor / 255.0 # Normalize to [0, 1]

        return undistorted_tensor

    @staticmethod
    def obs_to_smolvla_state(obs):
        """Converts observation to observation Tensor 0-padded for SmolVLA input"""
        return torch.tensor([obs, 0, 0, 0, 0, 0]).unsqueeze(0)
    
    @staticmethod
    def action_to_angle(action):
        """Converts SmolVLA output action to 1-DoF servo angle (rounded)"""
        return round(action.squeeze(0).tolist()[0])
