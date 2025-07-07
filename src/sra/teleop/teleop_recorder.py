import cv2
import numpy as np
from pathlib import Path
import time
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame
from sra.camera_handler import CameraHandler
from sra.servo_controller import ServoController
from sra.helpers import convert_angle_to_vla
from sra.aria_common import ctrl_c_handler

BASE_DELTA = 1 # Original per-press step
ACCELERATION = 7.5 # Degrees per second of hold
MAX_DELTA = 15 # Clamp so it can't run away
KEY_RESET_TIME = 0.1 # If you pause >100ms, treat as a new press

TARGET_FPS = 60 # Higher update rate for smoother movement
FRAME_TIME = 1.0 / TARGET_FPS

class TeleopRecorder:
    """Teleoperation system that records data in LeRobot format"""

    def __init__(self, arduino_port, baud_rate, streaming_interface, update_iptables, profile_name, device_ip, \
                 dataset_repo_id, dataset_root="./recorded_datasets", fps=10, resume=True):
        self.dataset_repo_id = dataset_repo_id
        self.resume = resume
        
        # Initialize servo components
        self.servo_controller = ServoController(arduino_port, baud_rate, init_angle=90)
        self.last_key = None
        self.key_press_time = 0.0
        self.key_last_time = 0.0
        self.key_held = False

        # Initialize camera components
        self.camera_handler = CameraHandler(streaming_interface, update_iptables, profile_name, device_ip)
        self.current_frame = None # For dataset input (toggles from existing to None to make sure duplicate frames aren't recorded)
        self.display_frame = None # Smoothes out display window

        # Create LeRobot dataset
        self.dataset = self.create_lerobot_dataset(dataset_repo_id, dataset_root, fps)
        self.dataset_changed = False # So we don't try saving when nothing was changed

        # Recording state
        self.recording = False
        self.awaiting_confirmation = False
        self.current_target_angle = 90
        self.current_task = ""
        self.episode_frame_count = 0

        # Display setup
        self.display_window = "Teleoperation Control"
        
        print("Teleoperation recorder initialized!")
        print("Controls:")
        print("  't' - Change task")
        print("  'r' - Start/stop recording episode")
        print("  'y' - Confirm and save episode")
        print("  'n' - Delete/discard episode")
        print("  '< (LEFT) / > (RIGHT)' - Move servo")
        print("  'q' or ESC - Quit")

    def create_lerobot_dataset(self, repo_id, root, fps):
        """
        Create LeRobot dataset with proper features for SmolVLA:
        - 1-DoF (0-padded)
        - 1 overhead camera (322x322x3 Aria)
        """

        dataset_path = Path(root) / repo_id

        # Check if dataset already exists
        if self.resume and dataset_path.exists() and (dataset_path / "meta" / "info.json").exists():
            print(f"Resuming existing dataset: {repo_id}")
            try:
                # Load existing dataset
                dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)
                print(f"Resumed dataset with {dataset.meta.total_episodes} existing episodes")
                return dataset
            except Exception as e:
                print(f"Failed to resume dataset: {e}")
                print("Creating new dataset instead...")
                self.cleanup()

        # Define features matching SmolVLA expectations (6 DoF with padding)
        features = {
            "observation.state": {
                "dtype": "float32",
                # "shape": [6], # 6 DoF format with padding
                "shape": (6,),
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            },
            "observation.image": {
                "dtype": "video",
                "shape": [322, 322, 3], # Aria processed image dimensions
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": int(fps),
                    "video.height": 322,
                    "video.width": 322,
                    "video.channels": 3,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "action": {
                "dtype": "float32",
                # "shape": [6], # 6 DoF format with padding
                "shape": (6,),
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
            }
        }

        try:
            # Create dataset
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=features,
                root=Path(root) / repo_id,
                robot_type="custom_aria_1dof",
                use_videos=True
            )
            print(f"Created new dataset")
            return dataset
        except Exception as e:
            print(f"Failed to initialize dataset: {e}")
            self.cleanup()
            return None
    
    def _prompt_for_task(self):
        """Prompt user for task input"""
        try:
            new_task = input("\nEnter task description: ").strip()
            self.current_task = new_task
            print(f"Task set: {new_task}")
        except EOFError:
            print("Task input cancelled")
    
    def start_recording(self):
        """Start recording a new episode"""
        # Start off each recording by prompting for the task
        if not self.current_task:
            self._prompt_for_task()
        
        self.recording = True
        self.awaiting_confirmation = False
        self.episode_frame_count = 0
        print(f"Started recording - Task: {self.current_task}")
        return True
    
    def stop_recording(self):
        """Stop recording and save episode"""
        if not self.recording:
            return
        
        self.recording = False
        self.awaiting_confirmation = True

        print(f"Recording stopped. Episode has {self.episode_frame_count} frames.")
        print("Press 'y' to confirm and save, or 'n' to discard this episode.")

        # # Save episode
        # self.dataset.save_episode()
        # self.dataset_changed = True
        # print(f"Episode saved with task: {self.current_task}")
    
    def confirm_episode(self):
        """Confirm and save the current episode"""
        if not self.awaiting_confirmation:
                return
            
        try:
            # Save episode to dataset
            self.dataset.save_episode()
            self.dataset_changed = True
            self.awaiting_confirmation = False
            print(f"sEpisode confirmed and saved! Task: {self.current_task}")
            print(f"Total episodes in dataset: {self.dataset.meta.total_episodes}")
        except Exception as e:
            print(f"Error saving episode: {e}")
            self.discard_episode()

    def discard_episode(self):
        """Discard the current episode and clear the buffer"""
        if not self.awaiting_confirmation:
            return
        
        try:
            # Clear the episode buffer without saving
            self.dataset.clear_episode_buffer()
            self.awaiting_confirmation = False
            self.episode_frame_count = 0
            print(f"Episode discarded. Buffer cleared.")
        except Exception as e:
            print(f"Error clearing buffer: {e}")
            self.cleanup()
            # # Fallback: try to reset the buffer manually
            # try:
            #     self.dataset.episode_buffer = {
            #         'size': 0,
            #         'observation.state': [],
            #         'observation.image': [],
            #         'action': []
            #     }
            #     self.awaiting_confirmation = False
            #     self.episode_frame_count = 0
            #     print("Buffer manually reset.")
            # except Exception as e2:
            #     print(f"Failed to reset buffer: {e2}")
    
    def record_frame(self):
        """Record a single frame if recording is active"""
        if not self.recording:
            return
        
        # Get current camera frame (if next frame hasn't come yet, don't record so we avoid duplicates)
        if self.current_frame is None:
            return
        
        # Get current servo position
        current_servo_angle = self.servo_controller.get_status()
        if current_servo_angle == -1:
            return
        
        # Convert angles from Arduino range [0, 180] to VLA range [-180, 180]
        servo_angle_vla = convert_angle_to_vla(current_servo_angle)
        target_angle_vla = convert_angle_to_vla(self.current_target_angle)

        # Create frame data with 6 DoF format (padding with zeros)
        frame_data = {
            "observation.state": np.array([servo_angle_vla, 0, 0, 0, 0, 0], dtype=np.float32),
            "observation.image": self.current_frame, # numpy array from camera
            "action": np.array([target_angle_vla, 0, 0, 0, 0, 0], dtype=np.float32)
        }

        # Add frame to dataset episode buffer
        self.dataset.add_frame(frame_data, task=self.current_task)
        self.episode_frame_count += 1
        self.current_frame = None
    
    def run_teleoperation(self):
        """Main teleoperation control loop"""
        cv2.namedWindow(self.display_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.display_window, 512, 512)
        cv2.setWindowProperty(self.display_window, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(self.display_window, 1200, 50)

        last_update = time.time()

        with ctrl_c_handler() as ctrl_c:
            while not (((key := (cv2.waitKey(1) & 0xFF)) in (27, ord('q'))) or ctrl_c):
                now = time.time()

                # Maintain consistent frame rate
                elapsed = now - last_update
                if elapsed < FRAME_TIME:
                    time.sleep(FRAME_TIME - elapsed)
                    now = time.time()
                last_update = now

                # Get and display current frame
                current_frame = self.camera_handler.get_processed_frame()
                if current_frame is not None:
                    self.current_frame = current_frame
                    self.display_frame = current_frame
                
                if self.display_frame is None:
                    key = cv2.waitKey(1) & 0xFF
                    continue
                
                # Create display frame with overlay information
                display_frame = self.create_display_frame(cv2.cvtColor(self.display_frame, cv2.COLOR_BGR2RGB))
                cv2.imshow(self.display_window, display_frame)

                # Handle control inputs
                if key == ord('r'): 
                    self.last_key = ord('r')
                    # Toggle recording:
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif key == ord('y') and self.awaiting_confirmation:
                    # Confirm episode
                    self.confirm_episode()
                elif key == ord('n') and self.awaiting_confirmation:
                    # Discard episode
                    self.discard_episode()
                elif key == ord('t') and not self.recording:
                    self.last_key = ord('t')
                    # Prompt for new task (only when not currently recording)
                    self._prompt_for_task()
                elif key in (ord(','), ord('.')):
                    # Movement keys
                    if key != self.last_key or (now - self.key_last_time) > KEY_RESET_TIME:
                        # New keypress
                        self.last_key = key
                        self.key_press_time = now
                        self.key_held = True
                        hold_time = 0.0
                    else:
                        # Continuing to hold the same key
                        hold_time = now - self.key_press_time

                    self.key_last_time = now

                    # Compute accelerated delta
                    delta = int(BASE_DELTA + ACCELERATION * hold_time)
                    delta = min(delta, MAX_DELTA)

                    # Apply movement
                    if key == ord(','): # Left ('<')
                        self.current_target_angle = max(0, self.current_target_angle - delta)
                    elif key == ord('.'): # Right ('>')
                        self.current_target_angle = min(180, self.current_target_angle + delta)

                    try:
                        self.servo_controller.move_servo(self.current_target_angle)
                    except Exception as e:
                        self.cleanup()
                        raise
                elif key == 255:
                    # Reset key state when no key is pressed
                    if self.key_held and (now - self.key_last_time) > KEY_RESET_TIME:
                        self.key_held = False
                        self.last_key = None
                
                # Record frame if recording is active
                self.record_frame()

        # Clean up
        if self.recording:
            self.stop_recording()
        elif self.awaiting_confirmation:
            # Auto-discard if exiting during confirmation
            self.discard_episode()
        
        self.cleanup()
        if self.dataset_changed:
            self.save_to_huggingface()
    
    def create_display_frame(self, frame):
        """Create display frame with overlay information"""
        display_frame = frame.copy()

        # Get current servo status
        current_servo_angle = self.servo_controller.get_status()

        # Add recording indicator
        if self.recording:
            cv2.putText(display_frame, "RECORDING", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Frames: {self.episode_frame_count}", (10, 70),  # MODIFIED
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.awaiting_confirmation:                      # NEW block
            cv2.putText(display_frame, "CONFIRM EPISODE", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Frames: {self.episode_frame_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, "Press 'y' to save, 'n' to discard", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add current task
        if self.current_task:
            cv2.putText(display_frame, f"Task: {self.current_task}", (10, 140),  # MODIFIED y-position
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add servo angle info
        if current_servo_angle != -1:
            cv2.putText(display_frame, f"Servo: {current_servo_angle} degrees", (10, 170),  # MODIFIED y-position
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Target: {self.current_target_angle} degrees", (10, 190),  # MODIFIED y-position
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame
    
    def save_to_huggingface(self):
        """Save dataaset to huggingface"""
        print("Saving and pushing dataset...")
        try:
            self.dataset.push_to_hub(
                private=False,
                push_videos=True, 
                tags=['lerobot', 'teleoperation', 'servo']
            )
            print(f"Dataset successfully pushed to: https://huggingface.co/datasets/{self.dataset.repo_id}")
        except Exception as e:
            print(f"Failed to push to hub: {e}")

    def cleanup(self):
        """Clean up resources and save dataset"""
        print("Cleaning up...")
        self.camera_handler.terminate()
        self.servo_controller.disconnect()
        cv2.destroyAllWindows()

        if self.dataset_changed:
            print(f"Dataset saved locally to: {self.dataset.root}")
