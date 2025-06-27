import serial
import time

class ServoController:
    """
    Class for Arduino communication to read/write servo angles.
    Initializes connection by sending "MARCO" to Arduino, expects "POLO" in response.
    - Movement commands take the form "MOVE:[angle]" and receives response "MOVED:[angle]"
    - Status request takes the form "STATUS" and receives response "STATUS:[angle]"
    - Sees angle range [0, +180]
    """
    def __init__(self, arduino_port, baud_rate, init_angle = 90, print_communications = False):
        # TODO: set up serial read/write timeout once proof of concept is working
            # 1, 2, 4, 8, 16 seconds (5 tries)

        self.print_communications = print_communications

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
        if self.print_communications:
            print(f"Moving servo to {angle}")
        self.arduino.write(command.encode())

        # Parses resulting angle from Arduino response as a float
        response = self.arduino.readline().decode().strip()
        try:
            response_angle = int(response[6:]) # After "MOVED:"
            if self.print_communications:
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
        if self.print_communications:
            print("Requesting servo status")

        # Parse resulting angle from Arduino
        response = self.arduino.readline().decode().strip()
        try:
            response_angle = int(response[7:]) # After "STATUS:"
            if self.print_communications:
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