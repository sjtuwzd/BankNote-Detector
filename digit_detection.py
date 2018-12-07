#!/usr/bin/env python
import cv2
import gzip
import matplotlib.pyplot as plt
import numpy as np
import struct
import sys
from scipy.linalg import eigh
import matplotlib.image as img
from scipy.misc import imread
from sklearn.preprocessing import normalize

import collections

def read_idx(mnist_filename):
    with gzip.open(mnist_filename) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I',
                      f.read(4))[0] for d in range(dims))
        data_as_array = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
        return data_as_array

def run(gray_in):
    mnist_train_images2 = read_idx("jpg_to_mnist/single-images-idx3-ubyte.gz")
    mnist_train_labels2 = read_idx("jpg_to_mnist/single-labels-idx1-ubyte.gz")

    number2 = mnist_train_images2.shape[0]
    height2 = mnist_train_images2.shape[1]
    width2 = mnist_train_images2.shape[2]
    X2 = np.ones((height2*width2, number2))

    for i in range(number2):
        norm_image2 = np.ones(height2*width2)
        norm_image2 = normalize(mnist_train_images2[i]) * 255
        image_pixel2 = (norm_image2).reshape(height2*width2)
        X2[:,i] = image_pixel2

    mean2 = X2.mean(1).reshape((height2*width2, 1))
    mean_matrix2 = np.tile(mean2, (1, number2))
    X2 = X2 - mean_matrix2
    XXT2 = np.matmul(X2, np.transpose(X2))
    w2, v2 = eigh(XXT2, eigvals=(764 ,783))

    W_train2 = np.ones((number2, 20))
    for i in range(number2):
        image_pixel_train2 = normalize(mnist_train_images2[i].reshape((1, height2*width2))) -mean2.reshape(1, height2*width2)
        W_train2[i] = np.matmul(image_pixel_train2, v2)
    
    im = gray_in
    im = cv2.resize(im, (465, 195))

    ret, im_th = cv2.threshold(im, 90, 255, cv2.THRESH_BINARY_INV)
    _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    single_digit = {}

    for rect in rects:
        # Detect a single digit: 1,2,5,0 and then connect
        if rect[2] < 75 and rect[3] < 80 and rect[2] > 10 and rect[3] > 20 : # rec[3] height
            roi = im_th[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            single_digit[rect] = {}
            single_digit[rect]["roi"] = roi
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
        image_pixel_test = normalize(single_digit[rect0]["roi"]).reshape(1, height2*width2) - mean2.reshape(1, height2*width2)
        W_test = np.matmul(image_pixel_test, v2)
        single_label = str(KNN(W_test[0], W_train2, mnist_train_labels2, mnist_train_images2))
        if len(digits) == 1:
            islegal = False
        if str(single_label) == "0":
            print("first digit is 0")
            islegal = False
        else:
            label += single_label
        for digit in digits[1:]:
            rect_remain = key_list[digit]
            image_pixel_test = normalize(single_digit[rect_remain]["roi"]).reshape(1, height2*width2) - mean2.reshape(1, height2*width2)
            W_test = np.matmul(image_pixel_test, v2)
            single_label = str(KNN(W_test[0], W_train2, mnist_train_labels2, mnist_train_images2))
            print(single_label)
            if str(single_label) != "0":
                print("remaining digit is not 0")
                islegal = False
            else:
                label += single_label
        if islegal == True:
            cv2.rectangle(im, (rect0[0], rect0[1]), (rect_remain[0] + rect_remain[2], rect_remain[1] + rect_remain[3]), (0, 255, 0), 3)
            cv2.putText(im, str(label), (rect0[0], rect0[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

    cv2.imshow("Resulting Image with Rectangular ROIs", im)
    cv2.waitKey()


def KNN(W_test, W_train, mnist_train_labels, mnist_train_images):
    distanceArray = np.ones(W_train.shape[0])

    for i in range(W_train.shape[0]):
        dis = calculateSquareDistance(W_test, W_train[i])
        distanceArray[i] = dis
    minPos = np.argsort(distanceArray)[:1]
    labels = mnist_train_labels[minPos]
    binCount = np.bincount(labels)
    label = np.argmax(binCount)
    return label

# It will return the square of distance
def calculateSquareDistance(p1, p2):
    diff = p1-p2
    return np.sum(diff**2)

def pre_process():
    path = input("Enter the file path : ").strip()
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (2,2))
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    filename = "{}.png".format("temp")
    return gray

if __name__ == "__main__":
    gray = pre_process()
    run(gray)
