import argparse

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
        default=115200,
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
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Huggingface dataset path"
    )
    parser.add_argument(
        "--model_path", type=str, default="lerobot/smolvla_base", help="Huggingface VLA model path"
    )
    return parser.parse_args()