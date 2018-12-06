#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from collections import defaultdict

import cv2
import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
import struct
import sys
from scipy.linalg import eigh
import scipy.misc
import matplotlib.image as img
import ast
from scipy.misc import imread
from dig_struct import *
from sklearn.preprocessing import normalize

import collections

# np.set_printoptions(threshold=np.nan)

def read_idx(mnist_filename):
    '''Reads both the MNIST images and labels.

    Args:
        mnist_filename: the path of the MNIST file

    Returns:
        data_as_array: a numpy array corresponding to the data within the
            MNIST file. For example, for MNIST images, the output is a
            (n, 28, 28) numpy array, where n is the number of images.
    '''
    with gzip.open(mnist_filename) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I',
                      f.read(4))[0] for d in range(dims))
        data_as_array = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
        return data_as_array

def run():

    mnist_train_images = read_idx("jpg_to_mnist/continuous-images-idx3-ubyte.gz")
    mnist_train_labels = read_idx("jpg_to_mnist/continuous-labels-idx1-ubyte.gz")
    mnist_train_images2 = read_idx("jpg_to_mnist/single-images-idx3-ubyte.gz")
    mnist_train_labels2 = read_idx("jpg_to_mnist/single-labels-idx1-ubyte.gz")

    number = mnist_train_images.shape[0]
    height = mnist_train_images.shape[1]
    width = mnist_train_images.shape[2]
    X = np.ones((height*width, number))

    number2 = mnist_train_images2.shape[0]
    height2 = mnist_train_images2.shape[1]
    width2 = mnist_train_images2.shape[2]
    X2 = np.ones((height2*width2, number2))

    for i in range(number):
        norm_image = np.ones(height*width)
        norm_image = normalize(mnist_train_images[i]) * 255
        image_pixel = (norm_image).reshape(height*width)
        X[:,i] = image_pixel

    for i in range(number2):
        norm_image2 = np.ones(height2*width2)
        norm_image2 = normalize(mnist_train_images2[i]) * 255
        image_pixel2 = (norm_image2).reshape(height2*width2)
        X2[:,i] = image_pixel2
    
    mean = X.mean(1).reshape((height*width, 1))
    mean_matrix = np.tile(mean, (1, number))
    X = X - mean_matrix
    XXT = np.matmul(X, np.transpose(X))
    w, v = eigh(XXT, eigvals=(764 ,783))

    mean2 = X2.mean(1).reshape((height2*width2, 1))
    mean_matrix2 = np.tile(mean2, (1, number2))
    X2 = X2 - mean_matrix2
    XXT2 = np.matmul(X2, np.transpose(X2))
    w2, v2 = eigh(XXT2, eigvals=(764 ,783))

    W_train = np.ones((number, 20))
    for i in range(number):
        image_pixel_train = normalize(mnist_train_images[i].reshape((1, height*width))) -mean.reshape(1, height*width)
        W_train[i] = np.matmul(image_pixel_train, v)
    
    W_train2 = np.ones((number2, 20))
    for i in range(number2):
        image_pixel_train2 = normalize(mnist_train_images2[i].reshape((1, height2*width2))) -mean2.reshape(1, height2*width2)
        W_train2[i] = np.matmul(image_pixel_train2, v2)
    
    im = cv2.imread("temp.png")
    im = cv2.resize(im, (465, 195))

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
    _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # continuous_digit_distance = []
    # continuous_digit_label = []
    # continuous_digit_rect = []
    # continuous_digit_roi = []

    single_digit = {}
    single_digit_label = []
    single_digit_rect = []
    single_digit_roi = []
    single_digit_rect_left_top = []

    for rect in rects:
        # Detect a continuous digit: 
        # if rect[2] < 75 and rect[3] < 65 and rect[2] > 30 and rect[3] > 30 : # rec[3] height
        #     roi = im_th[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        #     # Resize the image
        #     roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        #     image_pixel_test = normalize(roi).reshape(1, height*width) - mean.reshape(1, height*width)
        #     W_test = np.matmul(image_pixel_test, v) # get one test data in "face space” coordinates:            
        #     label, distance = KNN(W_test[0], W_train, mnist_train_labels, mnist_train_images)
        #     print(distance)
        #     continuous_digit_distance.append(distance)
        #     continuous_digit_label.append(label)
        #     continuous_digit_roi.append(roi)
        #     continuous_digit_rect.append(rect)
        # Detect a single digit: 1,2,5,0 and then connect
        if rect[2] < 75 and rect[3] < 80 and rect[2] > 10 and rect[3] > 20 : # rec[3] height
        # if True:
            roi = im_th[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            image_pixel_test = normalize(roi).reshape(1, height2*width2) - mean2.reshape(1, height2*width2)
            W_test = np.matmul(image_pixel_test, v2) # get one test data in "face space” coordinates:            
            label, distance = KNN(W_test[0], W_train2, mnist_train_labels2, mnist_train_images2)
            single_digit[rect] = {}
            # single_digit[rect]["left_point"] = rect[0]
            single_digit[rect]["label"] = label
            single_digit[rect]["roi"] = roi
            # cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            # cv2.imshow("roi_{0}".format(rect), roi)
            # cv2.putText(im, str(label), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    # print(single_digit)
    single_digit = collections.OrderedDict(sorted(single_digit.items()))
    key_list = []
    for key in single_digit:
        key_list.append(key)

    digits = []
    skip_value = -1
    if key_list:
        # find left digit
        for i in range(len(key_list)):
            if i < skip_value:
                continue
            else:
                rect = key_list[i]
                right_x = rect[0] + rect[2]
                right_top_y = rect[1]
                right_bottom_y = rect[1] + rect[3]
            for j in range(len(key_list[i+1:])):
                rect2 = key_list[j+i+1]
                left_x = rect2[0]
                left_top_y = rect2[1]
                left_bottom_y = rect2[1] + rect2[3]
                if abs(right_x - left_x) < 6 and abs(right_top_y-left_top_y) < 6 and abs(right_bottom_y-left_bottom_y)<6:
                    print("find nearby points")
                    skip_value = j+i+1
                    if i not in digits:
                        digits.append(i)
                    if j+i+1 not in digits:
                        digits.append(j+i+1)
                    break
    print(digits)
    # check if it is a correct digit
    islegal = True
    label = ""
    if digits:
        rect0 = key_list[digits[0]]
        if len(digits) == 1:
            islegal = False
        if str(single_digit[rect0]["label"]) == "0":
            print("first digit is 0")
            islegal = False
        else:
            label += str(single_digit[rect0]["label"])
        for digit in digits[1:]:
            rect_remain = key_list[digit]
            if str(single_digit[rect_remain]["label"]) != "0":
                print("remaining digit is not 0")
                islegal = False
            else:
                label += str(single_digit[rect_remain]["label"])
        if islegal == True:
            cv2.rectangle(im, (rect0[0], rect0[1]), (rect_remain[0] + rect_remain[2], rect_remain[1] + rect_remain[3]), (0, 255, 0), 3)
            # cv2.imshow("roi_{0}".format(rect), roi)
            cv2.putText(im, str(label), (rect0[0], rect0[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

            # im2 = cv2.imread("original.png")
            # im2 = cv2.resize(im2, (465, 195))
            # cv2.rectangle(im2, (rect0[0], rect0[1]), (rect_remain[0] + rect_remain[2], rect_remain[1] + rect_remain[3]), (0, 255, 0), 3)
            # cv2.putText(im2, str(label), (rect0[0], rect0[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
            # cv2.imshow("original figure", im2)

    cv2.imshow("Resulting Image with Rectangular ROIs", im)
    cv2.waitKey()


def KNN(W_test, W_train, mnist_train_labels, mnist_train_images):
    distanceArray = np.ones(W_train.shape[0])

    for i in range(W_train.shape[0]):
        dis = calculateSquareDistance(W_test, W_train[i])
        distanceArray[i] = dis
    minPos = np.argsort(distanceArray)[:1]
    labels = mnist_train_labels[minPos]
    # cv2.imshow("xxxx_{0}: ".format(minPos), mnist_train_images[minPos].reshape(28, 28))
    binCount = np.bincount(labels)
    label = np.argmax(binCount)
    return label, distanceArray[minPos]

# It will return the square of distance
def calculateSquareDistance(p1, p2):
    diff = p1-p2
    return np.sum(diff**2)


if __name__ == "__main__":
    run()
