import serial
import time

CMD_HANDSHAKE = 0x00
RSP_HANDSHAKE = 0xFF
CMD_MOVE = 0x01
CMD_STATUS = 0x02
RSP_MOVED = 0x81
RSP_STATUS = 0x82

class ServoController:
    """
    Class for Arduino communication to read/write servo angles. Sees angle range [0, +180]
    """
    #! Change for Higher-DOF: [cmd, arg_1, ..., arg_n]
    def __init__(self, arduino_port, baud_rate, init_angle = 90):
        # Connect to Arduino
        self.ser = serial.Serial(arduino_port, baud_rate, timeout=0.1)
        print("Connecting to Arduino...")
        time.sleep(2) # Wait 2 seconds to let Arduino reset

        # Perform handshake
        if not self._handshake():
            self.disconnect()
            raise IOError("Failed to handshake with Arduino")
        
        print(f"Connected to port {arduino_port} at {baud_rate} baud")
        # Move to initial position
        self.move_servo(init_angle)

    def _handshake(self, retries=5):
        """Send handshake request and expect RSP_HANDSHAKE back. Return success status as boolean"""
        for attempt in range(retries):
            self.ser.write(bytes([CMD_HANDSHAKE, 0x00]))
            resp = self.ser.read(2)
            if len(resp) == 2 and resp[0] == RSP_HANDSHAKE:
                return True
            time.sleep(0.1 * (2 ** attempt))
        return False
    
    def disconnect(self):
        """Close connection with Arduino"""
        print("Disconnecting from Arduino")
        self.ser.close()

    def _send(self, cmd_id, arg=0):
        """Send two-byte packet and read two-byte response."""
        self.ser.write(bytes([cmd_id, arg]))
        resp = self.ser.read(2)
        if len(resp) != 2:
            self.disconnect()
            raise IOError("Timeout or incomplete response")
        return resp[0], resp[1]
    
    def move_servo(self, angle):
        """
        Instructs Arduino to move servo to the given input angle. Returns the angle the Arduino received
        """
        angle = int(round(angle))
        if not 0 <= angle <= 180:
            self.disconnect()
            raise ValueError("Angle out of range")
        rsp_id, rsp_angle = self._send(CMD_MOVE, angle)
        if rsp_id != RSP_MOVED or rsp_angle != angle:
            self.disconnect()
            raise IOError(f"Bad MOVE response: {rsp_id:#02x}, {rsp_angle}")
        return rsp_angle
    
    def get_status(self):
        """Requests servo position from Arduino. If the response angle is an int within the servo bounds, return the angle as an int"""
        rsp_id, rsp_angle = self._send(CMD_STATUS, 0)
        if rsp_id != RSP_STATUS or not (0 <= rsp_angle <= 180):
            self.disconnect()
            raise IOError(f"Bad STATUS response: {rsp_id:#02x}, {rsp_angle}")
        return rsp_angle
