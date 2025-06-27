from sra.teleop.teleop_recorder import TeleopRecorder
from sra.teleop.cli import parse_args

def main():
    args = parse_args()

    recorder = TeleopRecorder(
        arduino_port=args.arduino_port,
        baud_rate=args.baud_rate,
        streaming_interface=args.streaming_interface,
        update_iptables=args.update_iptables,
        profile_name=args.profile_name,
        device_ip=args.device_ip,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root
    )

    recorder.run_teleoperation()

main()