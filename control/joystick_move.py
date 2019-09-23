import rospy
import sys
import os
import pygame
from geometry_msgs.msg import Twist

os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Acelerar el motor
def acelerar(pedal, rpm):
    if rpm < pedal*6.0: # desaceleracion 6.0 es tope
        rpm += pedal * 0.0001 # acelear poco

    if rpm > 1.0: # desaceleracion del motor
        rpm -= 0.00001
    return rpm


def move():
    # Starts a new node
    rospy.init_node('robot_cleaner', anonymous=True)
    velocity_publisher = rospy.Publisher('/catvehicle/cmd_vel_safe', Twist, queue_size=10)
    vel_msg = Twist()

    #Receiveing the user's input
    print("Let's move your robot")
    cambio = 0
    velocidad = 0.0
    rpm = 0.0
    sw = 0

    print "Cambio:", cambio
    while not rospy.is_shutdown():

        rpm = acelerar(abs(1 - joystick.get_axis(2))/2, rpm)
        freno = (abs(1 - joystick.get_axis(3)) -3.0517578125e-05)

        clutch = abs(1 - joystick.get_axis(1))/2
        if abs(velocidad) < (rpm * abs(cambio))/3:
            velocidad += (rpm * 0.0001) * cambio
            if clutch < 0.6:
                if sw == 1:
                    sw = 0
                    rpm = 3
                if sw == -1:
                    sw = 0
                    rpm = 6


        if velocidad > 0.0:
            velocidad -= 0.0001 * (10-rpm) + (freno * 0.1)
            rpm -= freno * 0.01
        else:
            velocidad += 0.00012 * (10-rpm) + (freno * 0.1)
            rpm += freno * 0.01

        # Transmision
        for event in pygame.event.get():
            vel_msg.angular.z = -joystick.get_axis(0)
            if event.type == pygame.JOYBUTTONDOWN:
                if clutch > 0.7:
                    # cambio -
                    if joystick.get_button(4) == 1:
                        if cambio < 5:
                            cambio += 1
                            sw = 1
                    # cambio +
                    elif joystick.get_button(5) == 1:
                        if cambio > -1:
                            cambio -= 1
                            sw = -1

                    print "Cambio:", cambio

        #print("acc: {%2.3f} " %velocidad, "rpm: {%2.3f}" %rpm, "steer: {%1.4f}" %vel_msg.angular.z)

        #Checking if the movement is forward or backwards
        vel_msg.linear.x = velocidad
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0

        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        #vel_msg.angular.z = 0

        velocity_publisher.publish(vel_msg)


if __name__ == '__main__':
    try:
        #Testing our function
        move()
    except rospy.ROSInterruptException: pass

#
