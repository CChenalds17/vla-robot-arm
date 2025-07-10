import serial
import time

CMD_HANDSHAKE = 0x00
RSP_HANDSHAKE = 0xFF
CMD_MOVE = 0x01
CMD_STATUS = 0x02
RSP_MOVED = 0x81
RSP_STATUS = 0x82

NUM_DOFS = 1

class ServoController:
    """
    Class for Arduino communication to read/write servo angles. Sees angle range [0, +180]
    """
    def __init__(self, arduino_port, baud_rate, init_angles=[90]*NUM_DOFS):
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
        self.move_servo(init_angles)

    def _handshake(self, retries=5):
        """Send handshake request and expect RSP_HANDSHAKE back. Return success status as boolean"""
        for attempt in range(retries):

            self.ser.write(bytes([CMD_HANDSHAKE] + [0x00] * NUM_DOFS))
            resp = self.ser.read(NUM_DOFS + 1)
            if len(resp) == NUM_DOFS + 1 and resp[0] == RSP_HANDSHAKE:
                return True
            time.sleep(0.1 * (2 ** attempt))
        return False
    
    def disconnect(self):
        """Close connection with Arduino"""
        print("Disconnecting from Arduino")
        self.ser.close()

    def _send(self, cmd_id, args=[0]*NUM_DOFS):
        """Send byte packet and read byte response."""
        self.ser.write(bytes([cmd_id] + args))
        resp = self.ser.read(NUM_DOFS + 1)
        if len(resp) != NUM_DOFS + 1:
            self.disconnect()
            raise IOError("Timeout or incomplete response")
        return resp[0], resp[1:]
    
    def move_servo(self, angles):
        """
        Instructs Arduino to move servos to the given input angles. Returns the angles the Arduino received
        """
        angles = [int(round(angle)) for angle in angles]
        for angle in angles:
            if not 0 <= angle <= 180:
                self.disconnect()
                raise ValueError("Angle out of range")
        rsp_id, rsp_angles = self._send(CMD_MOVE, angles)
        rsp_angles = [int(rsp_angle) for rsp_angle in rsp_angles]
        if rsp_id != RSP_MOVED or rsp_angles != angles:
            self.disconnect()
            raise IOError(f"Bad MOVE response: {rsp_id:#02x}, {rsp_angles}")
        return rsp_angles
    
    def get_status(self):
        """Requests servo position from Arduino. If the response angle is an int within the servo bounds, return the angle as an int"""
        rsp_id, rsp_angles = self._send(CMD_STATUS, [0]*NUM_DOFS)
        rsp_angles = [int(rsp_angle) for rsp_angle in rsp_angles]
        if rsp_id != RSP_STATUS:
            self.disconnect()
            raise IOError(f"Bad STATUS response: {rsp_id:#02x}, {rsp_angles}")
        for rsp_angle in rsp_angles:
            if not (0 <= rsp_angle <= 180):
                self.disconnect()
                raise IOError(f"Bad STATUS response: {rsp_id:#02x}, {rsp_angles}")
        return rsp_angles
