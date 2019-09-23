import rospy
import pickle
import cv2, sys, os, csv
import numpy as np
import matplotlib.pyplot as plt
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud, PointCloud2
from geometry_msgs.msg import Twist, TwistStamped
import sensor_msgs.point_cloud2 as pc2
import message_filters

from time import time
import fusion

rgbd = np.full((10, 20, 4), 0)
depth = np.full((10, 20, 3), 0)
rgb = []
lidar = []
bridge = CvBridge()
start_time = time()


record = True # Guardando datos desde ros?
record_count = 0
record_path = '../database/data/'

steering = 0.0
velocity = 0.0


def record_steer(vel):
    global steering, velocity
    steering = vel.angular.z
    velocity = vel.linear.x


def f():
    if fusion.bl.drv == None:
        print "iniciando CUDA context"
        drv.init()
        fusion.bl.drv = drv
        fusion.bl.dev = fusion.bl.drv.Device(0)

    global lidar, rgb, record_count, record_path, steering, velocity
    if len(lidar) == 0 or len(rgb) == 0:
        return
    #rgbd, rgb, depth = fusion.fusion_view(lidar, rgb)
    rgbd = fusion.fusion_view(lidar, rgb)

    #cv2.imshow("Fusion", rgb)
    #cv2.imshow("Imagen de profundidad", depth)
    #cv2.waitKey(50)
    if record and (steering != 0.0 and velocity != 0.0):
        st, vl = steering, velocity
        print "Steering:", st, "Vel:", vl
        pickle.dump(rgbd, open(record_path+str(record_count)+".p", "wb"))
        with open(record_path+'steering.csv', mode='a') as employee_file:
            employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            employee_writer.writerow([record_path+str(record_count)+".p",
                float(st),
                float(vl)])
        record_count += 1


def callback(image_raw, pcl):
    global rgb, lidar, start_time
    rgb = bridge.imgmsg_to_cv2(image_raw, "bgr8")
    lidar_points = []
    for p in pc2.read_points(pcl, field_names = ("x", "y", "z"), skip_nans=True):
        #print " x : %f  y: %f  z: %f" %(p[0],p[1],p[2])
        lidar_points.append(p[0])
        lidar_points.append(p[1])
        lidar_points.append(p[2])
    lidar = np.asarray(lidar_points)
    try:
        f()
    except:
        print "except:", sys.exc_info()[0]

    elapsed_time = time() - start_time
    print("Topic publish time: %0.10f seconds." % elapsed_time)
    print "----------------"
    start_time = time()


def listener():
    rospy.init_node('listener', anonymous=True)
    image_sub = message_filters.Subscriber("/catvehicle/camera_front/image_raw_front", Image)
    pcl_sub = message_filters.Subscriber("/catvehicle/lidar_points", PointCloud2)
    rospy.Subscriber("/catvehicle/cmd_vel_safe", Twist, record_steer)

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, pcl_sub], 10, 0.2)
    ts.registerCallback(callback)
    rospy.spin()


if __name__ == '__main__':
    if record:
        input_files = os.listdir(record_path)
        if 'steering.csv' in input_files:
            record_count = len(input_files) - 1
        else:
            record_count = len(input_files)

        print "Elements:", record_count

    try:
        listener()
    except rospy.ROSInterruptException:
        pass


#
