import time 
import serial
import sys
import Jetson.GPIO as GPIO
import numpy as np

#GOAL:
#	Establish a wireless serial connection between RPi and Nano
#	Transmit and receive matrices of different sizes
#	Compare the transmission times of each sizes

#Create serial
ser = serial.Serial(
	port = '/dev/ttyTHS1',
	baudrate = 115200,
	parity = serial.PARITY_NONE,
	stopbits = serial.STOPBITS_ONE,
	bytesize = serial.EIGHTBITS,
	timeout = .5 #Depending on the size of the matrix, the timout may need to be changed, larger timeout for larger matrices 
	)



Matrix = []
matrixempty = 1
narray = []
zero_count = 0
time_start = time.time()
#Size of the array expected to be transmitted in this case its a 27x27 matrix
#other arraylen:
#	27: 27x27
#	128: 128x128
#	144: 256x144
#	360: 480x360
arraylen = 27 


Zero_Matrix = list(np.zeros((arraylen), int))

#Receive messages
def receiver():
	global matrixempty, Matrix, narray, zero_count, time_start
	while ser.inWaiting() > 0:
		#Receive incoming message
		vect = ser.read(size=arraylen) #Change depending on the size of matrix data we are using 
		if vect == b'Image': #Tells the receiver that the matrix is complete and wait for the next message
			image()
			break
		if vect == b'Finish': #Tells the receiver that all matrices are complete and that there are no more messages to be received
			finish()
		
		print(vect)
		rdata=list(vect)
		print(rdata)
		print(len(rdata))
		#Check for any missing bytes
		if len(rdata) < len(Zero_Matrix):
			rarray = Zero_Matrix #replace array with missing bytes to an array of zeros 
			zero_count = zero_count + 1 #counts how many lines are replaced
		else:
			rarray = np.array([rdata]) #if nothing is missing, continue with the code
		
		print(rarray) 
		if matrixempty: #if matrix is empty add array to empty matrix
			Matrix = rarray
			matrixempty = 0
			print('Matrix')
			print(Matrix)
		else: #if not empty, stack the array to the matrix
			#Matrix = np.concatenate((Matrix, rarray), axis = 0)
			Matrix = np.vstack((Matrix, rarray)) 		
			print('Matrix') #show the matrix
			print(Matrix)

#matrix is finished, save matrix into a file for further instruction and clear the matrix to receive the next one
#only used if multiple matrices are being transmitted back to back
def image():
	global matrixempty, Matrix, narray, zero_count, time_start
	datafile = open('data.txt', 'a')
	datafile.write(str(Matrix))
	datafile.write('\n')
	datafile.write('\n')
	datafile.close()
	Matrix = []
	matrixempty = 1
	receiver()

#all matrices are received, tell the code to stop execution and calcualte the total time it took to receive and create the matrix
def finish():
	global matrixempty, Matrix, narray, zero_count, time_start
	time_end = time.time()
	time_total = time_end - time_start
	correct_lines = arraylen-zero_count #
	accuracy = correct_lines/arraylen
	#print('Accuracy:',accuracy)
	print('Total Time:', time_total)
	sys.exit()

while 1:
	receiver()



