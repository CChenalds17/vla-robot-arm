from cli import parse_args
from smolvla_sra_system import SmolVLASRASystem

def main():
    args = parse_args()

    # TODO: Dynamically select device (mps/cuda/cpu)

    system = SmolVLASRASystem(args.arduino_port, args.baud_rate, args.streaming_interface, \
                              args.update_iptables, args.profile_name, args.device_ip)
    
    system.run_control_loop()

if __name__ == "__main__":
    main()