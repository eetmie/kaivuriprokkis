import serial
import time


PORT = "COM7"
MULTIPLIER = 6

rotation = 0 # initial rotation we start from

ser = serial.Serial(PORT, 115200, timeout=1)
time.sleep(0.001)


default_rpm = f"m20\n"
default_acceleration = f"a5\n"
safety_stop_command = f"s\n"

ser.write(default_rpm.encode())
time.sleep(0.001)
ser.write(default_acceleration.encode())
time.sleep(0.001)

# ....

current_angle = ser.read()





while True:
    try:
        user_input = float(input("input rotation command: "))

        command = (user_input / 360) * MULTIPLIER

        final_command= f"r{command}\n"

        ser.write(final_command.encode())

        print(f"Command sent: {final_command}")
        
        rotation += command

        print(f"current rotation: {rotation}")

    except KeyboardInterrupt: # ctrl+c

        ser.write(safety_stop_command.encode())
        print("script killed safely :)")
        break



