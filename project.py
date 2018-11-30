import gzip
import matplotlib.pyplot as plt
import numpy as np
import struct
import sys
import os
import cv2 as cv
import operator
import collections

def sift():
    # for i in range(4):
    #     name = "data/test" + str(i) + ".jpg"
    #     print(name)
    #     original_Image = cv.imread(name)
    #     descriptor = cv.xfeatures2d.SIFT_create(4000)
    #     kp_original_surf, des_original_surf = descriptor.detectAndCompute(original_Image, None)
    #     bf_surf = cv.BFMatcher()
    #
    #     for eachphoto in os.listdir("data/england"):
    #         image = cv.imread("data/england/" + eachphoto)
    #         kp_curr_surf, des_curr_surf = descriptor.detectAndCompute(image, None)
    #
    #         matches_surf_middle = bf_surf.knnMatch(des_original_surf, des_curr_surf, k=1)
    #         matches_surf_count = 0
    #         for m in matches_surf_middle:
    #             if m[0].distance < 200:
    #                 matches_surf_count += 1
    #         print(matches_surf_count)
    #         if matches_surf_count > 190:
    #             print(eachphoto)

    for i in range(7):
        name = "data/test" + str(i) + ".jpg"
        print(name)
        original_Image = cv.imread(name)
        descriptor = cv.xfeatures2d.SIFT_create(4000)
        kp_original, des_original = descriptor.detectAndCompute(original_Image, None)
        flann = cv.BFMatcher()

        for eachphoto in os.listdir("data/england"):
            image = cv.imread("data/england/" + eachphoto)
            kp_curr, des_curr = descriptor.detectAndCompute(image, None)

            matches_middle = flann.knnMatch(des_original, des_curr, k=2)
            matches_count = 0
            # Apply ratio test
            for m, n in matches_middle:
                if m.distance < 0.5 * n.distance:
                    matches_count += 1

            print(matches_count)
            if matches_count > 50:
                print(eachphoto)


def FREAK():
    for i in range(7):
        name = "data/test" + str(i) + ".jpg"
        print(name)
        original_Image = cv.imread(name)
        descriptor = cv.xfeatures2d.FREAK_create()
        kp_original, des_original = descriptor.compute(original_Image, None)
        flann = cv.BFMatcher()

        for eachphoto in os.listdir("data/england"):
            image = cv.imread("data/england/" + eachphoto)
            kp_curr, des_curr = descriptor.compute(image, None)

            matches_middle = flann.knnMatch(des_original, des_curr, k=2)

            matches = sorted(matches_middle, key=lambda x: x.distance)
            img3 = cv.drawMatchesKnn(original_Image, kp_original, image, kp_curr, matches, None, flags=2)
            plt.imshow(img3), plt.show()


def BRISK():
    for i in range(7):
        name = "data/test" + str(i) + ".jpg"
        print(name)
        original_Image = cv.imread(name)
        descriptor = cv.BRISK_create()
        kp_original, des_original = descriptor.detectAndCompute(original_Image, None)
        flann = cv.BFMatcher()

        for eachphoto in os.listdir("data/england"):
            image = cv.imread("data/england/" + eachphoto)
            kp_curr, des_curr = descriptor.detectAndCompute(image, None)

            matches_middle = flann.knnMatch(des_original, des_curr, k=2)
            matches_count = 0
            # Apply ratio test
            for m, n in matches_middle:
                if m.distance < 0.75 * n.distance:
                    matches_count += 1

            print(matches_count)
            if matches_count > 50:
                print(eachphoto)


def FAST():
    for i in range(7):
        name = "data/test" + str(i) + ".jpg"
        print(name)
        original_Image = cv.imread(name)
        # Initiate FAST object with default values
        fast = cv.FastFeatureDetector_create()

        # find and draw the keypoints
        kp = fast.detect(original_Image, None)
        img2 = cv.drawKeypoints(original_Image, kp, color=(255, 0, 0), outImage=None)
        plt.imshow(img2), plt.show()


def main():
    # sift()
    # FREAK()
    # BRISK()
    FAST()


if __name__ == "__main__":
    main()