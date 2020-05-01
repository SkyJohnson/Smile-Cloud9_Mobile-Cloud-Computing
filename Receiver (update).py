import time 
import serial
import sys
import Jetson.GPIO as GPIO
import numpy as np

ser = serial.Serial(
	port = '/dev/ttyTHS1',
	baudrate = 115200,
	parity = serial.PARITY_NONE,
	stopbits = serial.STOPBITS_ONE,
	bytesize = serial.EIGHTBITS,
	timeout = 5
	)
#counter = 1
#vect = bytearray()
#Zero_Matrix = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Zero_Matrix = list(np.zeros((144), int))

Matrix = []
matrixempty = 1
narray = []
zero_count = 0
time_start = time.time()


while 1:
	time_start1 = time.time()
	while ser.inWaiting() > 0:
		vect = ser.read(size=144)
		time_start2 = time.time()
		rec = time_start2 - time_start1
		#print('Recieve time:', (rec))
		if vect == b'Finish':
			time_end = time.time()
			time_total = time_end - time_start
			correct_lines = 256-zero_count
			accuracy = correct_lines/256
			#print('Accuracy:',accuracy)
			print('Total Time:', time_total)
			sys.exit()
		print(vect)
		rdata=list(vect)
		print(rdata)
		
		print(len(rdata))
		if len(rdata) < len(Zero_Matrix):
			rarray = Zero_Matrix
			zero_count = zero_count + 1
		else:
			rarray = np.array([rdata])
		
		print(rarray) 
		if matrixempty:
			Matrix = rarray
			matrixempty = 0
			print('Matrix')
			print(Matrix)
		else:
			#Matrix = np.concatenate((Matrix, rarray), axis = 0)
			Matrix = np.vstack((Matrix, rarray))
			#if len(Matrix)			
			print('Matrix')
			print(Matrix)

		#rdata.clear()

		#x = ser.readline()
		#if x == '1':
			#print('Message Recieved %s\n'%(x))
			#y = int(x) + 1
			#print('New Value: %d' %(y))
			#newValue = '\nNew Value: %d\n' %(y)
			#ser.write(str.encode(newValue))
			#time.sleep(1)
		#else:
			#print(x)
			#reply = 'Recieved'
			#ser.write(str.encode(reply))
			#time.sleep(1)
