import rospy
import sys
import os
import pygame
from geometry_msgs.msg import Twist

pygame.init()
pygame.display.set_mode((100, 100))

def move():
    # Starts a new node
    rospy.init_node('robot_cleaner', anonymous=True)
    velocity_publisher = rospy.Publisher('/catvehicle/cmd_vel_safe', Twist, queue_size=10)
    vel_msg = Twist()

    #Receiveing the user's input
    print("Let's move your robot")

    velocidad = 0.0
    steering = 0.0
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        keys=pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            #print steering
            if steering > -1.0:
                steering -= 0.0002
        if keys[pygame.K_LEFT]:
            if steering < 1.0:
                steering += 0.0002

        if keys[pygame.K_UP]:
            if velocidad < 5.0:
                velocidad += 0.0001
        if keys[pygame.K_DOWN]:
            if velocidad > -1.0:
                velocidad -= 0.0002

        if keys[pygame.K_SPACE]:
            velocidad = 0.0

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    print("Autopilot desactivado")

        if velocidad > 0.0:
            velocidad -= 0.00005
        else:
            velocidad += 0.00005

        if steering > 0.0:
            steering -= 0.0001
        else:
            steering += 0.0001

        #print "vel", velocidad, "steer:", steering
        #Checking if the movement is forward or backwards
        vel_msg.linear.x = velocidad
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0

        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = steering

        velocity_publisher.publish(vel_msg)
        #rate.sleep()


if __name__ == '__main__':
    try:
        #Testing our function
        move()
    except rospy.ROSInterruptException: pass






#
