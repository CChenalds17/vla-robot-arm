import sys
import numpy as np
import cv2

import aria.sdk as aria
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord

# Camera constants
ARIA_ROI_LOWER_BOUND = 95
ARIA_ROI_UPPER_BOUND = 417

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
        # rgb_image = cv2.cvtColor(self.observer.rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = self.observer.rgb_image
        # Apply undistortion correction
        undistorted_rgb_image = distort_by_calibration(rgb_image, self.dst_calib, self.rgb_calib)
        # Rotate image
        undistorted_display = np.rot90(undistorted_rgb_image, -1)
        undistorted_display = undistorted_display[ARIA_ROI_LOWER_BOUND:ARIA_ROI_UPPER_BOUND, ARIA_ROI_LOWER_BOUND:ARIA_ROI_UPPER_BOUND]

        self.observer.rgb_image = None

        return undistorted_display