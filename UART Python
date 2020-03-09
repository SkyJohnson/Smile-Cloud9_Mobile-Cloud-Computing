import time
import serial
from time import sleep
import Jetson.GPIO as GPIO

ser = serial.Serial(
	port = "/dev/ttyTHS1",
	baudrate = 115200,
	parity = serial.PARITY_NONE,
	stopbits = serial.STOPBITS_ONE,
	bytesize = serial.EIGHTBITS,
	timeout=3
)
time.sleep(1)

while True:
	
	if ser.inWaiting() > 0:
		data = ser.read()
		print(data)
		ser.write(data)
		if data == "\n".encode():
			ser.write("\n".encode())
	#received_data = ser.readline()
	#received_data += ser.readline(ser.inWaiting())
	#print(received_data)
	#ser.write(received_data)
