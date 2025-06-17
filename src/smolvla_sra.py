import argparse
import sys
import cv2
import numpy as np
import serial
import time
import torch
import numpy as np
from PIL import Image

import aria.sdk as aria
from common import ctrl_c_handler, quit_keypress
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord

from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Camera constants
ARIA_ROI_LOWER_BOUND = 95
ARIA_ROI_UPPER_BOUND = 417

# Servo constants
INIT_ANGLE = 0

class CameraHandler:
    """Class to handle Aria glasses video feed"""

    def __init__(self, streaming_interface, update_iptables, profile_name, device_ip):
        self.setup(streaming_interface, update_iptables, profile_name, device_ip)
    
    def setup(self, streaming_interface, update_iptables, profile_name, device_ip):
        """Connect to Aria glasses, configure and calibrate sensors, and subscribe to Aria's RGB stream"""
        if update_iptables and sys.platform.startswith("linux"):
            update_iptables()
        
        #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
        aria.set_log_level(aria.Level.Info)

        # 1. Create DeviceClient instance, setting the IP address if specified
        self.device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        if device_ip:
            client_config.ip_v4_address = device_ip
        self.device_client.set_client_config(client_config)

        # 2. Connect to the device
        self.device = self.device_client.connect()

        # 3. Retrieve the device streaming_manager and streaming_client
        self.streaming_manager = self.device.streaming_manager
        self.streaming_client = self.streaming_manager.streaming_client

        # 4. Use a custom configuration for streaming
        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = profile_name
        # Note: by default streaming uses Wifi
        if streaming_interface == "usb":
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
        streaming_config.security_options.use_ephemeral_certs = True
        self.streaming_manager.streaming_config = streaming_config

        # 5. Get sensors calibration
        sensors_calib_json = self.streaming_manager.sensors_calibration()
        sensors_calib = device_calibration_from_json_string(sensors_calib_json)
        self.rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
        self.dst_calib = get_linear_camera_calibration(512, 512, 150, "camera-rgb")

        # 6. Start streaming
        self.streaming_manager.start_streaming()

        # 7. Configure subscription to listen to Aria's RGB stream
        config = self.streaming_client.subscription_config
        config.subscriber_data_type = aria.StreamingDataType.Rgb
        self.streaming_client.subscription_config = config

        # 8. Create and attach the visualizer and start listening to streaming data
        class StreamingClientObserver:
            def __init__(self):
                self.rgb_image = None
            
            def on_image_received(self, image: np.array, record: ImageDataRecord):
                self.rgb_image = image
        
        self.observer = StreamingClientObserver()
        self.streaming_client.set_streaming_client_observer(self.observer)
        self.streaming_client.subscribe()
    
    def terminate(self):
        """Unsubscribe from data and stop streaming"""
        print("Stop listening to Aria image data")
        self.streaming_client.unsubscribe()
        self.streaming_manager.stop_streaming()
        self.device_client.disconnect(self.device)
    
    def get_processed_frame(self):
        """Get processed frame for display windows and SmolVLA input. Returns undistorted_processed_image"""
        if self.observer.rgb_image is None:
            return None
        
        # Convert color space
        rgb_image = cv2.cvtColor(self.observer.rgb_image, cv2.COLOR_BGR2RGB)
        # Apply undistortion correction
        undistorted_rgb_image = distort_by_calibration(rgb_image, self.dst_calib, self.rgb_calib)
        # Rotate image
        undistorted_display = np.rot90(undistorted_rgb_image, -1)
        undistorted_display = undistorted_display[ARIA_ROI_LOWER_BOUND:ARIA_ROI_UPPER_BOUND, ARIA_ROI_LOWER_BOUND:ARIA_ROI_UPPER_BOUND]

        self.observer.rgb_image = None

        return undistorted_display

class ServoController:
    """
    Class for Arduino communication to read/write servo angles.
    Initializes connection by sending "MARCO" to Arduino, expects "POLO" in response.
    - Movement commands take the form "MOVE:[angle]" and receives response "MOVED:[angle]"
    - Status request takes the form "STATUS" and receives response "STATUS:[angle]"
    - Sees angle range [0, +180]
    """
    def __init__(self, arduino_port, baud_rate, init_angle):
        # TODO: set up serial read/write timeout once proof of concept is working
            # 1, 2, 4, 8, 16 seconds (5 tries)

        # Connect to Arduino
        self.arduino = serial.Serial(arduino_port, baud_rate)
        print("Connecting to Arduino...")
        time.sleep(2) # Wait 2 seconds to allow time for connection
        if self.confirm_connection():
            print(f"Connected to port {arduino_port} at {baud_rate} baud")
            self.move_servo(init_angle)
        else:
            print(f"Connection failed.")
            self.disconnect()
        
    def confirm_connection(self):
        """Confirms connection with Arduino through Marco Polo message. Returns connection status as boolean"""
        # Instruction
        self.arduino.write(b"MARCO\n")
        # Response
        response = self.arduino.readline().decode().strip()
        return response == "POLO"
    
    def move_servo(self, angle):
        """
        Instructs Arduino to move servo to the given input angle.
        Returns the angle the Arduino received and sets self.current_angle if it is equal to the transmitted angle.
        Otherwise, returns -1 and closes the connection
        """
        # Validate angle is within servo bounds
        angle = round(angle)
        if angle < 0 or angle > 180:
            print("Error: Angle is not in the valid range")
            self.disconnect()
            return -1
        
        # Send MOVE instruction to Arduino
        command = f"MOVE:{angle}\n"
        print(f"Moving servo to {angle}")
        self.arduino.write(command.encode())

        # Parses resulting angle from Arduino response as a float
        response = self.arduino.readline().decode().strip()
        try:
            response_angle = int(response[6:]) # After "MOVED:"
            print(f"Response: {response_angle}")
            if response_angle == angle:
                return response_angle
            else:
                print("Error: Response angle does not match")
        except ValueError:
            print("Error: Arduino response is not a valid integer")

        self.disconnect()
        return -1

    def get_status(self):
        """Requests servo position from Arduino. If the response angle is an int within the servo bounds, return the angle as a float. Otherwise, return -1"""
        # Send STATUS request to Arduino
        self.arduino.write(b"STATUS\n")
        print("Requesting servo status")

        # Parse resulting angle from Arduino
        response = self.arduino.readline().decode().strip()
        try:
            response_angle = int(response[7:]) # After "STATUS:"
            print(f"Status: {response_angle}")
            if 0 <= response_angle and response_angle <= 180:
                return response_angle
            else:
                print("Error: Angle is not in the valid range")
        except ValueError:
            print("Error: Arduino response is a not a valid integer")

        self.disconnect()
        return -1
    
    def disconnect(self):
        print("Disconnecting from Arduino")
        self.arduino.close()

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arduino_port",
        type=str,
        default="/dev/cu.usbmodem11101",
        help="Programming port of connected Arduino to allow Serial communication"
    )
    parser.add_argument(
        "--baud",
        dest="baud_rate",
        type=int,
        default=11520,
        help="Baud rate for Serial communication with Arduino. This needs to match the baud rate in the Arduino code.",
    )
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        default="usb",
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux",
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # TODO: Dynamically select device (mps/cuda/cpu)

    system = SmolVLASRASystem(args.arduino_port, args.baud_rate, args.streaming_interface, \
                              args.update_iptables, args.profile_name, args.device_ip)
    
    system.run_control_loop()

if __name__ == "__main__":
    main()