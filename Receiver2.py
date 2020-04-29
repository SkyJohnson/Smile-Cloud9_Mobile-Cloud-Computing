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
	timeout = 0
	)
counter = 1
vect = bytearray()
#Zero_Matrix = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Zero_Matrix = list(np.zeros((144), int))

#Matrix = []
matrixempty = 1
narray = []
time_start = time.time()
while 1:
		time_start1 = time.time()
	while ser.inWaiting():
		#Write the info into a file
		with open('datafile.dat', 'wb') as datafile:
			vect = ser.read(144)
			if vect == b'Finish':
				time_end = time.time()
				time_total = time_end - time_start
				print(time_total)
				break
			rdata = list(vect)
    			datafile.write(rdata)
			
			
		#vect= ser.read(144)
		time_start2 = time.time()
		rec = time_start2 - time_start1
		print('Recieve time:', (rec))
		
		
		
		#if vect == b'Finish':
		#	time_end = time.time()
		#	time_total = time_end - time_start
		#	print(time_total)
		#	sys.exit()
		
		#print(vect)
		#rdata=list(vect)
		#print(rdata)
		
		#print(len(rdata))
		#if len(rdata) < len(Zero_Matrix):
		#	rarray = Zero_Matrix
		#else:
		#	rarray = np.array([rdata])
		
		#print(rarray) 
		#if matrixempty:
		#	Matrix = rarray
		#	matrixempty = 0
		#	print('Matrix')
		#	print(Matrix)
		#else:
		#	#Matrix = np.concatenate((Matrix, rarray), axis = 0)
		#	Matrix = np.vstack((Matrix, rarray))
		#	#if len(Matrix)			
		#	print('Matrix')
		#	print(Matrix)

df = pd.read_csv('datsfile.dat', sep=",", header=None)	
		
#print(vect)
#rdata=list(vect)
#print(rdata)
		
#print(len(rdata))
#if len(rdata) < len(Zero_Matrix):
#	rarray = Zero_Matrix
#else:
#	rarray = np.array([rdata])
		
#print(rarray) 
#if matrixempty:
#	Matrix = rarray
#	matrixempty = 0
#	print('Matrix')
#	print(Matrix)
#else:
	#Matrix = np.concatenate((Matrix, rarray), axis = 0)
#	Matrix = np.vstack((Matrix, rarray))
#	#if len(Matrix)			
#	print('Matrix')
#	print(Matrix)
