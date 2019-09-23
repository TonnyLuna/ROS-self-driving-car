import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import time
import cv2
import csv
import math
import numpy as np

import interpolation as bl

lidar_width = 150
lidar_height = 80
points = lidar_width * lidar_height

lidar_hangle = 90.0#180.0
camera_hangle = 90.0
lidar_vangle = 33.678424
camera_vangle = 54.0

camera_width = 640
camera_height = 320

max_depth = 100.0
f = (camera_width/2.0) / math.tan( math.radians(camera_hangle)/2.0)


def load_image(data_dir):
    return mpimg.imread(data_dir)


def read_points(selectedIndex, path):
    i = 0
    lidarData = []
    with open(path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            i += 1
            if i == 1 or i is not selectedIndex:
                continue
            lidarData = np.asarray(row[4:points*3 + 4]).astype(np.float)
    csvFile.close()
    return lidarData


def to_image(vector):
    im = np.empty(lidar_width * lidar_height)
    j=0
    for i in range(len(vector)-3, 0, -3):
        im[j] = float(vector[i])
        j += 1
    print im
    return np.reshape(im, (lidar_height, -1))


def lidar_projection(vector):
    #image = np.full((lidar_height, int(lidar_width*3.129)), -1, dtype=np.float32)
    image = np.full((int(lidar_height), int(lidar_width*2.5)), -1, dtype=np.float32)
    h_c = np.zeros(len(vector)/3)
    v_c = np.zeros(len(vector)/3)
    h_count = 0.0
    v_count = 0.0
    j = 0
    for i in range(len(vector)-3, 0, -3):
        row = ((i/3)/lidar_width);
        col = (i/3)%lidar_width
        if col == 0:
            h_count = 0.0
        h_c[j] = int(h_count)
        v_c[j] = int(v_count)
        #h_count += math.ceil(abs((lidar_width/2-col)/280.0))+1
        h_count += math.ceil(abs((lidar_width/2-col)/40.0))+1
        if row > lidar_height/2:
            v_count = float(lidar_height - row) - (math.ceil(abs((lidar_height/2-row)/15.0)) * math.ceil(abs((lidar_width/2-col)/30.0)))
        else:
            v_count = float(lidar_height - row)
        j += 1

    image = bl.projection_cuda(image, vector.astype(np.float32), h_c.astype(np.int32), v_c.astype(np.int32),
        np.int32([image.shape[1]]), image.shape[0])

    return image


def to_3d(matrix):
    return np.repeat(matrix[:, :, np.newaxis], 3, axis=2).astype('uint8')


def view(im1, im2):
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(im1)

    plt.subplot(212)
    plt.imshow(im2)
    #plt.draw()
    #plt.pause(2)
    plt.show()


def cut(data, bias, vcut):
    min = data.shape[1]/4 - bias
    max = data.shape[1] - data.shape[1]/4 + bias

    return data[vcut:data.shape[0]-vcut, min:max]


def image_cut(data, xcut):
    inicio = int(abs((camera_height / (camera_vangle/lidar_vangle) - camera_height)/2))
    fin = data.shape[0] - int(abs((camera_height / (camera_vangle/lidar_vangle) - camera_height)/2))
    return data[inicio:fin, xcut:data.shape[1]-xcut, :]


def translation(img, x, y):
    translation_matrix = np.float32([ [1,0,x], [0,1,y] ])
    img_translation = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
    return img_translation


def fuse(lidar, image):
    img = image.copy()
    img = np.dstack((img, lidar))
    return img.astype('uint8')

def chanfle(lidar, image):
    img = image.copy()
    img[:, :, 0] = lidar[:, :]
    img[:, :, 1] = 0.0
    return img.astype('uint8')


def fusion_view(lidarVector, image):
    #lidarImage = lidarImage[:, 1730:3910]
    #start_time = time()

    image = image_cut(image, 0)
    lidarImage = lidar_projection(lidarVector)
    lidarImage = bl.median_interpolate_cuda(lidarImage, 1, 1, kernel_size=3)
    lidarImage = lidarImage[:, 5:370]
    lidarImage = cv2.resize(lidarImage, (640, 200))
    lidarImage = translation(lidarImage, 0, -10)

    #elapsed_time = time() - start_time
    #print("Fusion time: %0.10f seconds." % elapsed_time)
    return fuse(lidarImage, image)
    #return fuse(lidarImage, image), chanfle(lidarImage, image), to_3d(lidarImage)


def test_once():
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    drv.init()
    bl.drv = drv
    bl.dev = drv.Device(0)

    image = load_image('/home/tonny/bagfiles/2019-06-13-05-28-14/IMG/frame0000.jpg')
    image = image_cut(image, 0)

    lidarData = read_points(2, '/home/tonny/bagfiles/2019-05-29-19-56-13/lidar.csv')

    ant = time()
    lidarImage = lidar_projection(lidarData)
    view(to_3d(lidarImage), image)
    print "Parcial 1:", time() - ant
    ant = time()
    lidarImage = bl.median_interpolate_cuda(lidarImage, 1, 1, kernel_size=3)
    print "Parcial 2:", time() - ant
    ant = time()

    lidarImage = lidarImage[:, 5:370]

    lidarImage = cv2.resize(lidarImage, (image.shape[1], image.shape[0]))
    lidarImage = translation(lidarImage, 0, -6)

    view(to_3d(lidarImage), image)
    #view(to_3d(lidarImage), fuse(lidarImage, image))


if __name__ == '__main__':
    test_once()




#
