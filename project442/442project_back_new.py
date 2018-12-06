import numpy as np
import cv2
import common
import sys, getopt
import wave
import pyaudio
import os


def get_value(name):
    value_pos = name.find("_")
    value = name[0: value_pos]
    rest1 = name[value_pos+1:]
    back_front_pos = rest1.find("_")
    back_front = rest1[:back_front_pos]
    rest2 = rest1[back_front_pos+1:]
    begin_pos = rest2.find("_")
    end_pos = rest2.find(".")
    country = rest2[begin_pos+1:end_pos]
    if back_front == "0":
        side = "front"
    else:
        side = "back"
    out = country + " " + side + " " + value
    return out


def sift(path):
    name = path

    MIN_POINT = 20
    chunk = 1024
    FLANN_INDEX_KDTREE = 5

    cv2.useOptimized()
    cap = cv2.VideoCapture(0)
    detector = cv2.xfeatures2d.SIFT_create()
    norm = cv2.NORM_L1
    matcher = cv2.BFMatcher(norm)
    # matcher = cv2.BFMatcher()

    # des_list = np.load("video_des_list_gray.npy")
    # name_list = np.load("video_name_list_gray.npy")
    # kp_list = np.load("video_kp_list_gray.npy")
    # image_list = np.load("video_image_list_gray.npy")

    des_list = []
    name_list = []
    kp_list = []
    img_list = []

    for eachphoto in os.listdir("all_money_new"):
        image = cv2.imread("all_money_new/" + eachphoto)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (300, 140))
        img_list.append(image)
        kp_curr, des_curr = detector.detectAndCompute(image, None)
        des_list.append(des_curr)
        name_list.append(get_value(eachphoto))
        kp_list.append(kp_curr)

    frame = cv2.imread(name)
    img2 = cv2.resize(frame, (1000, 750))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    keep_continue = True

    # Capture frame-by-frame
    h1, w1 = 140, 300
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    find_one = False
    count = 0
    print("finished")
    while keep_continue:
        print(count)
        if count > 5:
            break
        keep_continue = False
        img1 = img2
        showText = ""
        p1 = []
        p2 = []
        for searchIndex in range(len(name_list)):
            img1_curr = img_list[searchIndex]
            kp1_curr = kp_list[searchIndex]
            desc1_curr = des_list[searchIndex]
            showText_curr = name_list[searchIndex]

            # calculate features
            kp2_curr, desc2_curr = detector.detectAndCompute(img2, None)
            # print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))

            # matching feature
            raw_matches = matcher.knnMatch(desc1_curr, trainDescriptors=desc2_curr, k=2)  # 2
            p1_curr, p2_curr, kp_pairs_curr = filter_matches(kp1_curr, kp2_curr, raw_matches)
            if len(p1_curr) > len(p1):
                img1 = img1_curr
                showText = showText_curr
                p1 = p1_curr
                p2 = p2_curr

        if len(p1) >= MIN_POINT:
            keep_continue = True
            if not find_one:
                find_one = True
                vis[:h2, w1:w1 + w2] = img2
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

            if count * h1 + h1 <= h2:
                vis[count * h1:h1 + count * h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            if H is not None:
                corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
                corners = np.int32(
                    cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
                cv2.polylines(vis, [corners], True, (0, 255, 0))
                cv2.fillPoly(img2, [corners - [300, 0]], (0, 0, 0))
                font = cv2.FONT_HERSHEY_SIMPLEX
                point1 = int((corners[0][0] + corners[2][0])/2)
                point2 = int((corners[0][1] + corners[2][1])/2)
                cv2.putText(vis, showText, (point1, point2), font, 1, (180, 0, 255), 2)

            for (x2, y2), inlier in zip(p2, status):
                if inlier:
                    col = (0, 255, 0)
                    cv2.circle(vis, (int(x2 + w1), int(y2)), 2, col, -1)
        count += 1
    cv2.imwrite("output.jpg", vis)
    cv2.imshow('find_obj', vis)
    cv2.waitKey(100000)


def filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    # print(mkp2)
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs


def video_detect():
    MIN_POINT = 30
    chunk = 1024
    FLANN_INDEX_KDTREE = 5

    cv2.useOptimized()
    cap = cv2.VideoCapture(0)
    detector = cv2.xfeatures2d.SIFT_create()
    norm = cv2.NORM_L1
    matcher = cv2.BFMatcher(norm)
    # matcher = cv2.BFMatcher()

    # des_list = np.load("video_des_list_gray.npy")
    # name_list = np.load("video_name_list_gray.npy")
    # kp_list = np.load("video_kp_list_gray.npy")
    # image_list = np.load("video_image_list_gray.npy")

    des_list = []
    name_list = []
    kp_list = []
    img_list = []

    for eachphoto in os.listdir("all_money_new"):
        image = cv2.imread("all_money_new/" + eachphoto)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (300, 140))
        img_list.append(image)
        kp_curr, des_curr = detector.detectAndCompute(image, None)
        des_list.append(des_curr)
        name_list.append(get_value(eachphoto))
        kp_list.append(kp_curr)

    # np.save("video_des_list_gray", des_list)
    # np.save("video_name_list_gray", name_list)
    # np.save("video_kp_list_gray", kp_list)
    # np.save("video_image_list_gray.npy", img_list)

    print("finished")
    while True:
        p = pyaudio.PyAudio()
        # switch template
        ret, frame = cap.read()
        img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keep_continue = True

        # Capture frame-by-frame
        h1, w1 = 140, 300
        h2, w2 = 480, 640
        vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
        find_one = False
        count = 0
        while keep_continue:
            count += 1
            keep_continue = False
            for searchIndex in range(len(name_list)):
                img1 = img_list[searchIndex]
                kp1 = kp_list[searchIndex]
                desc1 = des_list[searchIndex]
                showText = name_list[searchIndex]

                # calculate features
                kp2, desc2 = detector.detectAndCompute(img2, None)
                # print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))

                if len(kp2) > 0:
                    # matching feature
                    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
                    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
                    if len(p1) >= MIN_POINT:
                        keep_continue = True
                        if not find_one:
                            find_one = True
                            vis[:h1, :w1] = img1
                            vis[:h2, w1:w1 + w2] = img2
                            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

                        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                        if H is not None:
                            corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
                            corners = np.int32(
                                cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
                            cv2.polylines(vis, [corners], True, (0, 255, 0))
                            cv2.fillPoly(img2, [corners - [300, 0]], (0, 0, 0))
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(vis, showText, (corners[0][0], corners[0][1]), font, 1, (0, 0, 255), 2)

                        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
                            if inlier:
                                col = (0, 255, 0)
                                cv2.circle(vis, (int(x1), int(y1)), 2, col, -1)
                                cv2.circle(vis, (int(x2 + w1), int(y2)), 2, col, -1)

                if keep_continue:
                    break
                if not keep_continue:
                    if not find_one:
                        vis[:h1, :w1] = img1
                        vis[:h2, w1:w1 + w2] = img2
                cv2.imshow('find_obj', vis)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # When everything done, release the capture

    # close PyAudio
    p.terminate()

    cap.release()
    cv2.destroyAllWindows()




def main():
    input_type = input('Choose a function, 1 for image detect, 2 for video detect: ')
    type_int = int(input_type)
    if type_int == 1:
        image_path = input('please input the path of the image: ')
        sift(image_path)
    elif type_int == 2:
        video_detect()
    else:
        print("Wrong arguments!")
        return -1


if __name__ == '__main__':
    main()
