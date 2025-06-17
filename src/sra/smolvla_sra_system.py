from servo_controller import ServoController
from camera_handler import CameraHandler

from aria_common import ctrl_c_handler, quit_keypress

import torch
import cv2
from PIL import Image

from lerobot.lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Servo constants
INIT_ANGLE = 0

class SmolVLASRASystem:
    """
    System integrating Aria glasses, SmolVLA, and Arduino. Treats angle range as [-180, +180] and **converts every angle before transmission with Arduino**.
    """
    def __init__(self, arduino_port, baud_rate, streaming_interface, update_iptables, profile_name, device_ip, \
                 datasest_path="danaaubakirova/svla_so100_task4_v3_clean", model_path="lerobot/smolvla_base", device="mps"):
        # Initialize components (parameters supplied by command-line arguments)

        self.device = device
        self.dataset = LeRobotDataset(datasest_path)
        self.policy = SmolVLAPolicy.from_pretrained(model_path)
        self.setup_pol_state_dict()

        self.servo_controller = ServoController(arduino_port, baud_rate, init_angle=SmolVLASRASystem.convert_angle_to_arduino(INIT_ANGLE))
        self.camera_handler = CameraHandler(streaming_interface, update_iptables, profile_name, device_ip)

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

    def run_control_loop(self):
        """Control loop to operate SRA"""
        # TODO: Implement multithreading for task retrieval, etc
            # I think I will need a task listener or something

        cv2.namedWindow(self.undistorted_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.undistorted_window, 512, 512)
        cv2.setWindowProperty(self.undistorted_window, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(self.undistorted_window, 1200, 50)

        with ctrl_c_handler() as ctrl_c:
            while not (quit_keypress() or ctrl_c):
                undistorted_display = self.camera_handler.get_processed_frame()

                key = cv2.waitKey(1)
                if undistorted_display is not None:
                    cv2.imshow(self.undistorted_window, undistorted_display)

                    # (For now) Upon pressing SPACE, generate prediction and send it to Arduino
                    if key == ord(' '):
                        observation = self.get_observation(undistorted_display)
                        action = self.policy.select_action(observation)

                        print(f"VLA Action: {action}")
                        servo_angle = SmolVLASRASystem.action_to_angle(action)
                        self.servo_controller.move_servo(SmolVLASRASystem.convert_angle_to_arduino(servo_angle))
        
        self.camera_handler.terminate()

    def get_observation(self, undistorted_display):
        obs = SmolVLASRASystem.convert_angle_to_vla(self.servo_controller.get_status()) # Gets servo angle from Arduino and converts it
        observation_state = SmolVLASRASystem.obs_to_smolvla_state(obs).to(self.device)
        observation_image_top = SmolVLASRASystem.img_to_smolvla_tensor(undistorted_display).to(self.device)
        task = self.get_task()

        observation = {
            "observation.state": observation_state,
            "observation.image": observation_image_top,
            "task": [task]
        }

        print('---------------------------------------------------------')
        print(f"Observation State: {observation_state}")
        SmolVLASRASystem.tensor_to_pil(observation_image_top).show()
        print(f"Observation Image Top: {observation_image_top}")
        print(f"Observation Image Top Shape: {observation_image_top.shape}")
        print(f"Task: {task}")
        print('---------------------------------------------------------')

        return observation

    def get_task(self):
        return "Move left"
        
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

    @staticmethod
    def convert_angle_to_arduino(vla_angle: int) -> int:
        """Maps angle from [-180, +180] to [0, +180]"""
        return int((vla_angle + 180) / 2.0)

    @staticmethod
    def convert_angle_to_vla(arduino_angle: int) -> int:
        """Maps angle from [0, +180] to [-180, +180]"""
        return int(2.0 * arduino_angle - 180)

    @staticmethod
    def tensor_to_pil(t: torch.Tensor, mode="RGB") -> Image.Image:
        t = t.cpu().detach().squeeze(0)
        arr = (t * 255).clamp(0,255).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(arr, mode=mode)