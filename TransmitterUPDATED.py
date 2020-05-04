import time
import serial
import random
import RPi.GPIO as GPIO
import numpy as np

ser = serial.Serial(
    port = '/dev/ttyS0',
    baudrate = 115200,
    parity = serial.PARITY_NONE,
    stopbits = serial.STOPBITS_ONE,
    bytesize = serial.EIGHTBITS,
    timeout = 20
    )


#DIFFERENT SIZES OF MATRICES FOR TESTING
#Main two dimensions used for the project
pic = np.random.randint(0,255, size=(27,27), dtype=np.uint8)
#pic = np.random.randint(0,255, size=(128,128), dtype=np.uint8)

#Test dimensions for comparisson 
#pic = np.random.randint(0,255, size=(256,144), dtype=np.uint8)
#pic = np.random.randint(0,255, size=(480,360), dtype=np.uint8)





time_start = time.time()
#print(pic)

#TO SEND MULTIPLE MATRICES BACK TO BACK INCREASE i 
i = 0

while i < 1:
    #SEND EACH MATRIX BY READING EACH ROW IN THE MATRIX AND TRASNMIST IT SERIALLY
    for row in pic:
        #list_row = list(row)
        data = bytearray(row)
        print(row)
        print(data)
        print(len(data))
        ser.write(data)
        time.sleep(.1)
    #Once the whole matrix is transmitted, tell the receiver that the image is done
    #The transmitted will know that the matrix is finished and that the next message will either be a 'Finish' message which stops receiving
    #or the next matrix is about to be sent
    ser.write(str.encode('Image'))
    time.sleep(1)
    i = i + 1
   

#Once all matrices are transmitted, tell the receiver that it is finished and no more matrices are to be transmitted
ser.write(str.encode('Finish'))

#Record timings of transmission/execution time
time_end = time.time()
time_tot= time_end - time_start
print(time_tot)


