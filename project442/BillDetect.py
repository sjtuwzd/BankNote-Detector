import numpy as np
import cv2
import common
import time
# import thread
import sys, getopt
import wave
import pyaudio
from common import anorm, getsize

# --------------------------------------------------------------------------------------------------------
MIN_POINT = 30
chunk = 1024
FLANN_INDEX_KDTREE = 5


def init_feature():
    detector = cv2.xfeatures2d.SIFT_create()
    norm = cv2.NORM_L1
    # http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/

    flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=MIN_POINT)
    matcher = cv2.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio=0.75):  # ratio = 0.75
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


def explore_match(img1, img2, p1,p2, status=None, H=None):
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (0, 255, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis, showText, (corners[0][0], corners[0][1]), font, 1, (0, 0, 255), 2)
    # if status is None:
        # status = np.ones(len(kp_pairs), np.bool_)
    # p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    # print(type(p1))
    # # p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)
    # print(w1)
    # p2 = np.int32([kpp[1].pt for kpp in kp_pairs])

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        # print(inlier)
        if inlier:
            col = (0, 255, 0)
            cv2.circle(vis, (int(x1), int(y1)), 2, col, -1)
            cv2.circle(vis, (int(x2+w1), int(y2)), 2, col, -1)
    return vis


# def match_and_draw(win):
#
#
#     return found


# --------------------------------------------------------------------------------------------------------
cv2.useOptimized()

cap = cv2.VideoCapture(0)
detector, matcher = init_feature()

checksound = True
found = False
searchIndex = 1
count = 0
#
# preLoad resource
img_source1 = cv2.imread('img_source/20f.bmp', 0)
temp_kp1, temp_desc1 = detector.detectAndCompute(img_source1, None)

img_source2 = cv2.imread('img_source/20b.bmp', 0)
img_source2 = cv2.resize(img_source2, (300, 157))
temp_kp2, temp_desc2 = detector.detectAndCompute(img_source2, None)

img_source3 = cv2.imread('img_source/50f.bmp', 0)
temp_kp3, temp_desc3 = detector.detectAndCompute(img_source3, None)

img_source4 = cv2.imread('img_source/50b.bmp', 0)
temp_kp4, temp_desc4 = detector.detectAndCompute(img_source4, None)

img_source5 = cv2.imread('img_source/100f.bmp', 0)
temp_kp5, temp_desc5 = detector.detectAndCompute(img_source5, None)

img_source6 = cv2.imread('img_source/100b.bmp', 0)
temp_kp6, temp_desc6 = detector.detectAndCompute(img_source6, None)

img_source7 = cv2.imread('img_source/5f.bmp', 0)
temp_kp7, temp_desc7 = detector.detectAndCompute(img_source7, None)

img_source8 = cv2.imread('img_source/5b.bmp', 0)
temp_kp8, temp_desc8 = detector.detectAndCompute(img_source8, None)

img_source9 = cv2.imread('img_source/USA1F.bmp', 0)
img_source9 = cv2.resize(img_source9, (300, 157))
temp_kp9, temp_desc9 = detector.detectAndCompute(img_source9, None)

img_source10 = cv2.imread('img_source/USA5F.bmp', 0)
img_source10 = cv2.resize(img_source10, (300, 157))
temp_kp10, temp_desc10 = detector.detectAndCompute(img_source10, None)

img_source13 = cv2.imread('img_source/USA10F.bmp', 0)
img_source13 = cv2.resize(img_source13, (300, 157))
temp_kp13, temp_desc13 = detector.detectAndCompute(img_source13, None)


while True:
    t1 = cv2.getTickCount()
    p = pyaudio.PyAudio()
    # switch template
    ret, frame = cap.read()
    img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keep_continue = True

    # Capture frame-by-frame
    h1, w1 = 157, 300
    h2, w2 = 480, 640
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    find_one = False
    count = 0
    while keep_continue:
        count += 1
        keep_continue = False
        for searchIndex in range(11, 14):
            if searchIndex == 1:
                img1 = img_source1
                kp1 = temp_kp1
                desc1 = temp_desc1
                showText = '20'
            elif searchIndex == 2:
                img1 = img_source2
                kp1 = temp_kp2
                desc1 = temp_desc2
                showText = '20'
            elif searchIndex == 3:
                img1 = img_source3
                kp1 = temp_kp3
                desc1 = temp_desc3
                showText = '50'
            elif searchIndex == 4:
                img1 = img_source4
                kp1 = temp_kp4
                desc1 = temp_desc4
                showText = '50'
            elif searchIndex == 5:
                img1 = img_source5
                kp1 = temp_kp5
                desc1 = temp_desc5
                showText = '100'
            elif searchIndex == 6:
                img1 = img_source6
                kp1 = temp_kp6
                desc1 = temp_desc6
                showText = '100'
            elif searchIndex == 7:
                img1 = img_source7
                kp1 = temp_kp7
                desc1 = temp_desc7
                showText = '500'
            elif searchIndex == 8:
                img1 = img_source8
                kp1 = temp_kp8
                desc1 = temp_desc8
                showText = '500'
            elif searchIndex == 9:
                img1 = img_source9
                kp1 = temp_kp9
                desc1 = temp_desc9
                showText = '1000'
            elif searchIndex == 10:
                img1 = img_source10
                kp1 = temp_kp10
                desc1 = temp_desc10
                showText = '1000'
            if searchIndex == 11:
                img1 = img_source11
                kp1 = temp_kp11
                desc1 = temp_desc11
                showText = 'USA-1 dollar front'
            elif searchIndex == 12:
                img1 = img_source12
                kp1 = temp_kp12
                desc1 = temp_desc12
                showText = 'USA-5 dollar front'
            else:
                img1 = img_source13
                kp1 = temp_kp13
                desc1 = temp_desc13
                showText = 'USA-10 dollar front'

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
                        # print(inlier)
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

    # t2 = cv2.getTickCount()
    # calculate fps
    # time = (t2 - t1) / cv2.getTickFrequency()
    # print('FPS = ', 1 / time)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything done, release the capture

# close PyAudio
p.terminate()

cap.release()
cv2.destroyAllWindows()



# import numpy as np
# import cv2
# import common
# import time
# # import thread
# import sys, getopt
# import wave
# import pyaudio
# from common import anorm, getsize
#
# # --------------------------------------------------------------------------------------------------------
# MIN_POINT = 30
# chunk = 1024
# FLANN_INDEX_KDTREE = 5
#
#
# def init_feature():
#     detector = cv2.xfeatures2d.SIFT_create()
#     norm = cv2.NORM_L1
#     # http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/
#
#     flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=MIN_POINT)
#     matcher = cv2.BFMatcher(norm)
#     return detector, matcher
#
#
# def filter_matches(kp1, kp2, matches, ratio=0.75):  # ratio = 0.75
#     mkp1, mkp2 = [], []
#     for m in matches:
#         if len(m) == 2 and m[0].distance < m[1].distance * ratio:
#             m = m[0]
#             mkp1.append(kp1[m.queryIdx])
#             mkp2.append(kp2[m.trainIdx])
#     p1 = np.float32([kp.pt for kp in mkp1])
#     p2 = np.float32([kp.pt for kp in mkp2])
#     # print(mkp2)
#     kp_pairs = zip(mkp1, mkp2)
#     return p1, p2, kp_pairs
#
#
# def explore_match(img1, img2, p1,p2, status=None, H=None):
#     vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
#     vis[:h1, :w1] = img1
#     vis[:h2, w1:w1 + w2] = img2
#     vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
#     if H is not None:
#         corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
#         corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
#         cv2.polylines(vis, [corners], True, (0, 255, 0))
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(vis, showText, (corners[0][0], corners[0][1]), font, 1, (0, 0, 255), 2)
#     # if status is None:
#         # status = np.ones(len(kp_pairs), np.bool_)
#     # p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
#     # print(type(p1))
#     # # p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)
#     # print(w1)
#     # p2 = np.int32([kpp[1].pt for kpp in kp_pairs])
#
#     for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
#         # print(inlier)
#         if inlier:
#             col = (0, 255, 0)
#             cv2.circle(vis, (int(x1), int(y1)), 2, col, -1)
#             cv2.circle(vis, (int(x2+w1), int(y2)), 2, col, -1)
#     return vis
#
#
# # def match_and_draw(win):
# #
# #
# #     return found
#
#
# # --------------------------------------------------------------------------------------------------------
# cv2.useOptimized()
#
# cap = cv2.VideoCapture(0)
# detector, matcher = init_feature()
#
# checksound = True
# found = False
# searchIndex = 1
# count = 0
#
# # preLoad resource
# # img_source1 = cv2.imread('img_source/20f.bmp', 0)
# # temp_kp1, temp_desc1 = detector.detectAndCompute(img_source1, None)
# #
# # img_source2 = cv2.imread('img_source/20b.bmp', 0)
# # img_source2 = cv2.resize(img_source2, (300, 157))
# # temp_kp2, temp_desc2 = detector.detectAndCompute(img_source2, None)
# #
# # img_source3 = cv2.imread('img_source/50f.bmp', 0)
# # temp_kp3, temp_desc3 = detector.detectAndCompute(img_source3, None)
# #
# # img_source4 = cv2.imread('img_source/50b.bmp', 0)
# # temp_kp4, temp_desc4 = detector.detectAndCompute(img_source4, None)
# #
# # img_source5 = cv2.imread('img_source/100f.bmp', 0)
# # temp_kp5, temp_desc5 = detector.detectAndCompute(img_source5, None)
# #
# # img_source6 = cv2.imread('img_source/100b.bmp', 0)
# # temp_kp6, temp_desc6 = detector.detectAndCompute(img_source6, None)
# #
# # img_source7 = cv2.imread('img_source/500f.bmp', 0)
# # temp_kp7, temp_desc7 = detector.detectAndCompute(img_source7, None)
# #
# # img_source8 = cv2.imread('img_source/500b.bmp', 0)
# # temp_kp8, temp_desc8 = detector.detectAndCompute(img_source8, None)
# #
# # img_source9 = cv2.imread('img_source/1000f.bmp', 0)
# # temp_kp9, temp_desc9 = detector.detectAndCompute(img_source9, None)
# #
# # img_source10 = cv2.imread('img_source/1000b.bmp', 0)
# # img_source10 = cv2.resize(img_source10, (300, 157))
# # temp_kp10, temp_desc10 = detector.detectAndCompute(img_source10, None)
#
# img_source11 = cv2.imread('img_source/USA1F.bmp', 0)
# img_source11 = cv2.resize(img_source11, (300, 157))
# temp_kp11, temp_desc11 = detector.detectAndCompute(img_source11, None)
#
# img_source12 = cv2.imread('img_source/USA5F.bmp', 0)
# img_source12 = cv2.resize(img_source12, (300, 157))
# temp_kp12, temp_desc12 = detector.detectAndCompute(img_source12, None)
#
# img_source13 = cv2.imread('img_source/USA10F.bmp', 0)
# img_source13 = cv2.resize(img_source13, (300, 157))
# temp_kp13, temp_desc13 = detector.detectAndCompute(img_source13, None)
#
# frame = cv2.imread("test3.jpg")
# img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# img2 = cv2.resize(img2, (1000, 800))
# keep_continue = True
#
# # Capture frame-by-frame
# h1, w1 = 157, 300
# h2, w2 = img2.shape[:2]
# vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
# find_one = False
# while keep_continue:
#     if count > 4:
#         break
#     count += 1
#     keep_continue = False
#     for searchIndex in range(11, 14):
#         # if searchIndex == 1:
#         #     img1 = img_source1
#         #     kp1 = temp_kp1
#         #     desc1 = temp_desc1
#         #     showText = '20'
#         # elif searchIndex == 2:
#         #     img1 = img_source2
#         #     kp1 = temp_kp2
#         #     desc1 = temp_desc2
#         #     showText = '20'
#         # elif searchIndex == 3:
#         #     img1 = img_source3
#         #     kp1 = temp_kp3
#         #     desc1 = temp_desc3
#         #     showText = '50'
#         # elif searchIndex == 4:
#         #     img1 = img_source4
#         #     kp1 = temp_kp4
#         #     desc1 = temp_desc4
#         #     showText = '50'
#         # elif searchIndex == 5:
#         #     img1 = img_source5
#         #     kp1 = temp_kp5
#         #     desc1 = temp_desc5
#         #     showText = '100'
#         # elif searchIndex == 6:
#         #     img1 = img_source6
#         #     kp1 = temp_kp6
#         #     desc1 = temp_desc6
#         #     showText = '100'
#         # elif searchIndex == 7:
#         #     img1 = img_source7
#         #     kp1 = temp_kp7
#         #     desc1 = temp_desc7
#         #     showText = '500'
#         # elif searchIndex == 8:
#         #     img1 = img_source8
#         #     kp1 = temp_kp8
#         #     desc1 = temp_desc8
#         #     showText = '500'
#         # elif searchIndex == 9:
#         #     img1 = img_source9
#         #     kp1 = temp_kp9
#         #     desc1 = temp_desc9
#         #     showText = '1000'
#         # elif searchIndex == 10:
#         #     img1 = img_source10
#         #     kp1 = temp_kp10
#         #     desc1 = temp_desc10
#         #     showText = '1000'
#         if searchIndex == 11:
#             img1 = img_source11
#             kp1 = temp_kp11
#             desc1 = temp_desc11
#             showText = 'USA-1 dollar front'
#         elif searchIndex == 12:
#             img1 = img_source12
#             kp1 = temp_kp12
#             desc1 = temp_desc12
#             showText = 'USA-5 dollar front'
#         else:
#             img1 = img_source13
#             kp1 = temp_kp13
#             desc1 = temp_desc13
#             showText = 'USA-10 dollar front'
#
#         # calculate features
#         kp2, desc2 = detector.detectAndCompute(img2, None)
#         # print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))
#
#         if len(kp2) > 0:
#             # matching feature
#             raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
#             p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
#             if len(p1) >= MIN_POINT:
#                 keep_continue = True
#                 if not find_one:
#                     find_one = True
#                     vis[:h1, :w1] = img1
#                     vis[:h2, w1:w1 + w2] = img2
#                     vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
#
#                 H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
#                 if H is not None:
#                     corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
#                     corners = np.int32(
#                         cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
#                     cv2.polylines(vis, [corners], True, (0, 255, 0))
#                     vis2 = np.copy(vis)
#                     print(corners)
#                     print(corners - [300, 0])
#                     cv2.fillPoly(img2, [corners - [300, 0]], (0, 0, 0))
#                     # cv2.imshow("test", img2)
#                     # cv2.waitKey(1000)
#                     font = cv2.FONT_HERSHEY_SIMPLEX
#                     cv2.putText(vis, showText, (corners[0][0], corners[0][1]), font, 1, (0, 0, 255), 2)
#
#                 for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
#                     # print(inlier)
#                     if inlier:
#                         col = (0, 255, 0)
#                         cv2.circle(vis, (int(x1), int(y1)), 2, col, -1)
#                         cv2.circle(vis, (int(x2 + w1), int(y2)), 2, col, -1)
#
#         if keep_continue:
#             break
#         if not keep_continue:
#             if not find_one:
#                 vis[:h1, :w1] = img1
#                 vis[:h2, w1:w1 + w2] = img2
# cv2.imshow('find_obj', vis)
# cv2.waitKey(100000)



