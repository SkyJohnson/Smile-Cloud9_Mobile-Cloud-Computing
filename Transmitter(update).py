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
counter = 0
sent = 0
recieved = 1

#pic = np.random.randint(0,255, size=(20,20), dtype=np.uint8)
pic = np.random.randint(0,255, size=(256,144), dtype=np.uint8)
#pic = np.random.randint(0,255, size=(480,360))
#pic = np.random.randint(0,255, size=(720,480))
#pic = np.random.randint(0,255, size=(1280,720))
#pic = np.random.randint(0,255, size=(1280,720))


#pic = [[random.randint(0,255) for j in range(0, num_rows)] for i in range(num_columns)]

time_start = time.time()
#print(pic)



for row in pic:
    #list_row = list(row)
    data = bytearray(row)
    print(row)
    print(data)
    print(len(data))
    #ser.write(str.encode(row))
    ser.write(data)
    time.sleep(.1)
ser.write(str.encode('Finish'))
    
time_end = time.time()

time_tot= time_end - time_start

print(time_tot)

#while 1:
   # ser.write(str.encode('Write counter: %d\n'%(counter)))
    #time.sleep(1)
    #counter += 1
    
#     ser.write(str.encode('1'))
#     x = ser.readline().decode()
#     print(x)
#     time.sleep(2)
#         
#     ser.write(str.encode('6'))
#     x = ser.readline().decode()
#     print(x)
#     time.sleep(2)



# 
#     data2 = bytearray(elements)
#     
#     ser.write(data2)
#     print(data2)
#     time.sleep(2)
#     
#     data2 = bytearray(elements1)
#     
#     ser.write(data2)
#     print(data2)
#     time.sleep(2)
#     
#     data2 = bytearray(elements2)
#     
#     ser.write(data2)
#     print(data2)
#     time.sleep(2)



#     if x == '1':
#         print('Message Recieved %s\n'%(x))
#         y = int(x) + 1
#         print('New Value: %d' %(y))
#         ser.write(str.encode('\n New Value %d\n'%(y)))
#         time.sleep(3)
#     else:
#         print('Message Recieved %s\n'%(x))
#         check = x.isdigit()
#         if check:
#             y = int(x) + 20
#             print('New Value: %d\n'%(y))
#             ser.write(str.encode('\n New Value %d\n'%(y)))
#         else:
#             ser.write(str.encode('\n Please send a number\n'))
#             print('Waiting Int')  
