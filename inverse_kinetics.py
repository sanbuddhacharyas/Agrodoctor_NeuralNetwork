import serial
import math
import os
import time

class inverse_kinematics():
    def __init__(self, port,baud ,timeout):
       # self.ser = serial.Serial(port, baud)
        self.l1 = 21
        self.l2 = 17.5
        self.kin = serial.Serial(port,baud)
        self.distance =0
        self.left_right=0
    
    def calculate_angle(self,angle):
        x = angle[0]
        y = angle[1]
        z = angle[2]
        converter = (180 / math.pi)
        b = math.sqrt(x*x + z*z)
        print("b===>"+str(b))
        q2 = (self.l1**2 + b**2 - self.l2**2) / (2 * self.l1 * b)
        q1 = math.atan2(z ,x)*converter
        print("q1"+str(q1))
        q2 = math.acos(q2)*converter
        print("q2===>"+str(q2))
        self.Q2 = q1 + q2
       
        Q = (self.l1**2 + self.l2**2 - b**2) / (2 * self.l1 * self.l2)
        Q = math.acos(Q)*converter
        self.Q1 = (180 - Q - self.Q2)
        Q3 = y / x
        self.Q3 = math.asin(Q3)*converter
        print("Q1===>"+str(self.Q1))
        print("Q2===>"+str(self.Q2))
        print("Q3===>"+str(self.Q3))

    def send_angle(self,distance):
        self.Q1 = str(round(self.Q1))+'p'#1st Arm
        self.kin.write(self.Q1.encode())
       
        

        self.Q2 = str(round(self.Q2))+'q'#2nd Arm
        self.kin.write(self.Q2.encode())
    
        
        self.Q3 = str(round(self.Q3))+'r'#Rotation
        self.kin.write(self.Q3.encode())
        

        self.distance = str(distance)+'d'#wheel_distance
        self.kin.write(self.distance.encode())
        

        # self.left_right = str(left)+'l'#wheel_distance
        # self.kin.write(self.left_right.encode())

    def send_angle_test(self,p ,q, r,distance,data):
        p = str(round(p))+'p'#1st Arm
        self.kin.write(p.encode())


        q = str(round(q))+'q'#1st Arm
        self.kin.write(q.encode())


        r = str(round(r))+'r'#Rotation
        self.kin.write(r.encode())


        distance = str(distance)+'d'#wheel_distance
        self.kin.write(distance.encode())

        if data == 1:
            self.kin.write(b'o')
            time.sleep(1)
            self.kin.write(b'f')
        else :
            self.kin.write(b'f')

       

        

        # self.left_right = str(left)+'l'#wheel_distance
        # self.kin.write(self.left_right.encode())


    
    def serial_close(self):
        self.kin.close()
        
    
# inverse = inverse_kinematics("/dev/rfcomm0",38400,1)
# a = 30
# b = 5
# c = 10
# while True:

#     a = input('p')
#     b = input('q')
#     c = input ('r')
#     d = input('d')
#     e = input('e')
#     a = int(a)
#     b = int(b)
#     c = int(c)
#     d = int(d)
#     e = int (e)
#    # angle = [a ,b ,c]
#    # inverse.calculate_angle(angle)
#     inverse.send_angle_test( a , b, c, d,e)
#     r = inverse.kin.read()
#     print(r)
   # inverse.kin.write(b'o')
