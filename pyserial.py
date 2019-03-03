import serial
import math

class inverse_kinematics():
    def __init__(self, port,baud ,timeout):
       # self.ser = serial.Serial(port, baud)
        self.l1 = 21
        self.l2 = 17.5
        
    
    def calculate_angle(self, x,y,z):
        b = math.sqrt(x*x + z*z)
        q2 = (self.l1^2 + b^2 - self.l2^2) / (2 * self.l1 * b)
        q1 = math.atan2(z ,x)
        q2 = math.acos(q2)
        Q1 = q1 + q2
        Q = (self.l1^2 + self.l2^2 - b^2) / (2 * self.l1 * self.l2)
        Q = math.acos(Q)
        Q2 = (180 - Q - Q1)
        Q3 = y / x
        Q2 = math.asin(Q3)

        return Q1, Q2, Q3

    