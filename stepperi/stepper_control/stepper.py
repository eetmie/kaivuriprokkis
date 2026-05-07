import serial #serial communication library, "pip install pyserial"
import time

# --- CHANGE THESE ---
PORT = "COM7"       # Change to your port (e.g. COM4, COM5 ...)
MULTIPLIER = 6      # <-- THIS IS WHAT YOU NEED TO FIGURE OUT

# --------------------

ser = serial.Serial(PORT, 115200, timeout=1)
time.sleep(1)  # wait the system for a bit, just to be sure!
#lets send slower rotate speed for testing:
initial_command = "m20" #max speed 20rpm
ser.write(initial_command.encode())

# we can also send the smoother acceleration command
accel_command = "a5"
ser.write(accel_command.encode())

def rotate(rotations):
    command = f"r{rotations}\n"
    ser.write(command.encode())

while True:
    degrees = float(input("How many degrees? "))
    rotations = (degrees / 360) * MULTIPLIER
    rotate(rotations)
