import argparse
import sys
import cv2
import numpy as np
import serial
import time
from collections import deque

import aria.sdk as aria
from common import ctrl_c_handler, quit_keypress, update_iptables
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord

# Camera constants
ARIA_ROI_LOWER_BOUND = 95
ARIA_ROI_UPPER_BOUND = 417

# Servo constants
INIT_ANGLE = 90

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
        print("Stop listening to Aria image data")
        self.streaming_client.unsubscribe()
        self.streaming_manager.stop_streaming()
        self.device_client.disconnect(self.device)
    
    def get_processed_frames(self):
        """Get processed frames for display windows and SmolVLA input. Returns (rgb_processed_image, undistorted_processed_image)."""
        if self.observer.rgb_image is None:
            return None, None
        
        # Convert color space
        rgb_image = cv2.cvtColor(self.observer.rgb_image, cv2.COLOR_BGR2RGB)
        # Apply undistortion correction
        undistorted_rgb_image = distort_by_calibration(rgb_image, self.dst_calib, self.rgb_calib)
        # Rotate image
        rgb_processed = np.rot90(rgb_image, -1)
        undistorted_processed = np.rot90(undistorted_rgb_image, -1)

        self.observer.rgb_image = None

        return rgb_processed, undistorted_processed[ARIA_ROI_LOWER_BOUND:ARIA_ROI_UPPER_BOUND, ARIA_ROI_LOWER_BOUND:ARIA_ROI_UPPER_BOUND]

class ServoController:
    """
    Class for Arduino communication to read/write servo angles.
    Initializes connection by sending "MARCO" to Arduino, expects "POLO" in response.
    - Movement commands take the form "MOVE:[angle]" and receives response "MOVED:[angle]"
    - Status request takes the form "STATUS" and receives response "STATUS:[angle]"
    """
    def __init__(self, arduino_port='/dev/cu.usbmodem1101', baud_rate=11520):
        # TODO: set up serial read/write timeout once proof of concept is working

        # Connect to Arduino
        self.arduino = serial.Serial(arduino_port, baud_rate)
        print("Connecting to Arduino...")
        time.sleep(2) # Wait 2 seconds to allow time for connection
        if self.confirm_connection():
            print(f"Connected to port {arduino_port} at {baud_rate} baud")
        else:
            print(f"Connection failed.")
            self.arduino.close()
        self.current_angle = self.move_servo(INIT_ANGLE)
    
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
        Otherwise, returns -1
        """
        # TODO: Make sure angle is within servo bounds
        # Send MOVE instruction to Arduino
        command = f"MOVE:{int(angle)}\n"
        print(f"Moving servo to {angle}")
        self.arduino.write(command.encode())

        # Parses resulting angle from Arduino response as an int
        response = self.arduino.readline().decode().strip()
        response_angle = int(response[6:])
        print(f"Response: {response_angle}")

        # Check that angles match
        if response_angle == angle:
            self.current_angle = angle
            return response_angle
        return -1

    def get_status(self):
        """Requests servo position from Arduino. If the response angle is an int within the servo bounds, return the angle. Otherwise, return -1"""
        # Send STATUS request to Arduino
        self.arduino.write(b"STATUS\n")
        print("Requesting servo status")

        # Parse resulting angle from Arduino
        response = self.arduino.readline().decode().strip()
        response_angle = response[7:]
        print(f"Status: {response_angle}")

        # TODO: Validate bounds of angle
        if response_angle.isdigit():
            return int(response_angle)
        return -1

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    controller = ServoController(arduino_port="/dev/cu.usbmodem11101")
    while True:
        command = input("Command: ")
        instruction = command.split(' ')
        if instruction[0].lower() == "move" and len(instruction) > 1:
            controller.move_servo(instruction[1])
        elif instruction[0].lower() == "status":
            controller.get_status()

    # args = parse_args()
    
    # camera_handler = CameraHandler(args.streaming_interface, args.update_iptables, args.profile_name, args.device_ip)

    # rgb_window = "Aria RGB"
    # undistorted_window = "Undistorted RGB"

    # cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(rgb_window, 512, 512)
    # cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    # cv2.moveWindow(rgb_window, 50, 50)

    # cv2.namedWindow(undistorted_window, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(undistorted_window, 512, 512)
    # cv2.setWindowProperty(undistorted_window, cv2.WND_PROP_TOPMOST, 1)
    # cv2.moveWindow(undistorted_window, 600, 50)

    # with ctrl_c_handler() as ctrl_c:
    #     while not (quit_keypress() or ctrl_c):
    #         orig, undistorted = camera_handler.get_processed_frames()
    #         if orig is not None and undistorted is not None:
    #             cv2.imshow(rgb_window, orig)
    #             cv2.imshow(undistorted_window, undistorted)

    # # 10. Unsubscribe from data and stop streaming
    # camera_handler.terminate()

if __name__ == "__main__":
    main()