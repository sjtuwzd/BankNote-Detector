import gzip
import matplotlib.pyplot as plt
import numpy as np
import struct
import sys
import os
import cv2 as cv
import operator
import collections

def orb():
    for i in range(1,14):
        name = "data/test/test" + str(i) + ".jpg"
        print(name)
        original_Image = cv.imread(name)
        gray_orig = cv.cvtColor(original_Image, cv.COLOR_BGR2GRAY)
        descriptor = cv.ORB_create()

        kp_original, des_original = descriptor.detectAndCompute(gray_orig, None)
        flann = cv.BFMatcher()

        max_name = ""
        max_match = 0

        for eachphoto in os.listdir("data/ans"):
            image = cv.imread("data/ans/" + eachphoto)
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            kp_curr, des_curr = descriptor.detectAndCompute(gray_image, None)

            matches_middle = flann.knnMatch(des_original, des_curr, k=2)
            matches_count = 0
            # Apply ratio test
            for m, n in matches_middle:
                if m.distance < 0.75 * n.distance:
                    matches_count += 1

            if (matches_count>max_match):
                max_name = eachphoto
                max_match = matches_count

        print("max_match:", max_match)
        print("max_name:", max_name)
        if(max_name == "test"+str(i)+"_ans.jpg"):
            print("Correct")
        else:
            print("Wrong")


def surf():
    for i in range(1,14):
        name = "data/test/test" + str(i) + ".jpg"
        print(name)
        original_Image = cv.imread(name)
        gray_orig = cv.cvtColor(original_Image, cv.COLOR_BGR2GRAY)
        descriptor = cv.xfeatures2d.SURF_create()

        kp_original, des_original = descriptor.detectAndCompute(gray_orig, None)
        flann = cv.BFMatcher()

        max_name = ""
        max_match = 0

        for eachphoto in os.listdir("data/ans"):
            image = cv.imread("data/ans/" + eachphoto)
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            kp_curr, des_curr = descriptor.detectAndCompute(gray_image, None)

            matches_middle = flann.knnMatch(des_original, des_curr, k=2)
            matches_count = 0
            # Apply ratio test
            for m, n in matches_middle:
                if m.distance < 0.75 * n.distance:
                    matches_count += 1

            if (matches_count>max_match):
                max_name = eachphoto
                max_match = matches_count

        print("max_match:", max_match)
        print("max_name:", max_name)
        if(max_name == "test"+str(i)+"_ans.jpg"):
            print("Correct")
        else:
            print("Wrong")


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
    # FAST()
    # surf()
    orb()


if __name__ == "__main__":
    main()