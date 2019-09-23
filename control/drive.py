import argparse, base64, shutil, time, sys, os
import rospy, pygame, subprocess, threading, socketio, eventlet
import eventlet.wsgi
import message_filters
import numpy as np
import pandas as pd
import tensorflow as tf
import pycuda.driver as drv
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pc2
from datetime import datetime
from flask import Flask
from keras.models import load_model
from keras import metrics, backend
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError

import time

import rnn_utils as utils
from rnn_utils import STEPS, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
import fusion


model = None
graph = tf.get_default_graph()


MAX_SPEED = 10
MIN_SPEED = 1
speed_limit = MAX_SPEED
steering_angle = 0.0
throttle = 0.0
speed = 0.0
IS_RECURRENT = True

# permite la conduccion manual
manual = False

# Cuenta el tiempo  para reactivar el Autopilot
manual_time = 0.0
inter_time = 0.0
test_time = 0.0
interrupciones = 0

os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

rospy.init_node('talker', anonymous=True)
pub = rospy.Publisher('/catvehicle/cmd_vel_safe', Twist, queue_size=10)
vel_msg = Twist()


images = np.empty([1, STEPS, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS], dtype=object)

# comandos emitidos por el modelo neuronal
steering_data = []

def reset():
    global manual_time, inter_time, test_time, interrupciones
    manual_time = 0.0
    inter_time = 0.0
    test_time = time.time()
    interrupciones = 0.0

# metricas
def conduction_conduct():
    std = np.std(steering_data)
    # std dev pista 1
    data_df1 = pd.read_csv("/home/tonny/UdacitySim/Udacity_Sim/metrics/ref1/driving_log.csv")
    data_df1['steering'] = pd.to_numeric(data_df1['steering'], errors='coerce')
    std_r1 = np.std(data_df1['steering'])
    # std dev pista 2
    data_df2 = pd.read_csv("/home/tonny/UdacitySim/Udacity_Sim/metrics/ref2/driving_log.csv")
    data_df2['steering'] = pd.to_numeric(data_df2['steering'], errors='coerce')
    std_r2 = np.std(data_df2['steering'])

    m1 = abs(std - std_r1)
    m2 = abs(std - std_r2)

    x = np.linspace(std, std, len(steering_data))
    plt.plot(steering_data, label='self-driving')
    plt.plot(x, label='std dev')
    plt.xlabel('Time (s/10)')
    plt.ylabel('Steering Angle (r)')

    if m1 < m2:
        print("Abs std dev dif:", m1)

        x2 = np.linspace(std_r1, std_r1, len(data_df1['steering']))
        plt.plot(data_df1['steering'], label='steering data ref')
        plt.plot(x2, label='std dev')

    else:
        print("abs std dev dif:", m2)

        x2 = np.linspace(std_r2, std_r2, len(data_df2['steering']))
        plt.plot(data_df2['steering'], label='steering data ref')
        plt.plot(x2, label='std dev')

    plt.legend()
    plt.show()


def time_step(images, image):
    for i in range(STEPS-1):
        images[0][i] = images[0][i+1]
    images[0][STEPS-1] = image
    return images


rgbd = np.full((10, 20, 4), 0)
depth = np.full((10, 20, 3), 0)
rgb = []
lidar = []
bridge = CvBridge()
start_time = time.time()

def f():
    if fusion.bl.drv == None:
        print "iniciando CUDA context"
        drv.init()
        fusion.bl.drv = drv
        fusion.bl.dev = fusion.bl.drv.Device(0)

    global lidar, rgb
    if len(lidar) == 0 or len(rgb) == 0:
        return
    #rgbd, rgb, depth = fusion.fusion_view(lidar, rgb)
    return fusion.fusion_view(lidar, rgb)


def callback(image_raw, pcl):
    global rgb, rgbd, lidar, start_time
    global graph

    rgb = bridge.imgmsg_to_cv2(image_raw, "bgr8")
    lidar_points = []
    for p in pc2.read_points(pcl, field_names = ("x", "y", "z"), skip_nans=True):
        #print " x : %f  y: %f  z: %f" %(p[0],p[1],p[2])
        lidar_points.append(p[0])
        lidar_points.append(p[1])
        lidar_points.append(p[2])
    lidar = np.asarray(lidar_points)
    try:
        rgbd = f()
    except:
        print "except:", sys.exc_info()[0]

    global steering_angle, throttle, interrupciones, images, speed
    try:
        image = np.asarray(rgbd).astype(np.float32)       # from PIL image to numpy array
        image = utils.preprocess(image) # apply the preprocessing
        image = np.array([image])       # the model expects 4D array
        images = time_step(images, image)

        # predict the steering angle for the image
        if not manual:
            with graph.as_default():
                # predict the steering angle for the image
                if IS_RECURRENT:
                    steering_angle = float(model.predict(images, batch_size=1))
                else:
                    steering_angle = float(model.predict(image, batch_size=1))

        global speed_limit
        if speed > speed_limit:
            speed_limit = MIN_SPEED  # slow down
        else:
            speed_limit = MAX_SPEED
        if not manual:
            throttle = (1.0 - steering_angle**2 - (speed/speed_limit)**2)*2

        # save the command for metrics
        if not np.isnan(steering_angle):
            steering_data.append(steering_angle)

        if np.isnan(speed):
            steering_angle = 0.0
        if np.isnan(speed):
            speed = 0.0
        if np.isnan(throttle):
            throttle = 0.0

        print "throttle:", throttle, "Steering:", steering_angle
        vel_msg.linear.x = throttle
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = steering_angle*-1

    except Exception as e:
        print(e)
    """elapsed_time = time.time() - start_time
    print("Topic publish time: %0.10f seconds." % elapsed_time)
    print "----------------"
    start_time = time.time()"""


def publish(clock):
    global vel_msg
    #speed = vel.linear.x
    pub.publish(vel_msg)


def thread_ejecute():
    # Feedback
    subprocess.call(['/home/tonny/joystick-1.4.7/utils/feedback', '-d', '/dev/input/event16'])


def thread_socket():
    import socket

    print("=========== socket ============ ")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        while True:
            mensaje = str(steering_angle)
            b = mensaje.encode()
            s.sendall(b)


if __name__ == '__main__':
    # ejecutar el feedback en c
    #cmd = threading.Thread(target=thread_ejecute)
    #socket = threading.Thread(target=thread_socket)
    #cmd.start()
    #socket.start()

    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()

    #model = load_model(args.model, custom_objects={'rmse': rmse, 'f1':f1, 'auc_roc':auc_roc})
    model = load_model(args.model)

    #test_time = time.time()

    image_sub = message_filters.Subscriber("/catvehicle/camera_front/image_raw_front", Image)
    pcl_sub = message_filters.Subscriber("/catvehicle/lidar_points", PointCloud2)
    rospy.Subscriber('/catvehicle/cmd_vel_safe', Twist, publish)

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, pcl_sub], 10, 0.2)
    ts.registerCallback(callback)
    rospy.spin()


#
