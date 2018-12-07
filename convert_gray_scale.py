
import cv2
import pytesseract
from PIL import Image
import operator
import os

def main():
    dirname = "./number_test_"
    print(os.listdir(dirname))
    for filename in os.listdir(dirname): # [1:] Excludes .DS_Store from Mac OS
        if filename.endswith(".png"):
            path = (os.path.join(dirname,filename))

            # Get File Name from Command Line
            # path = input("Enter the file path : ").strip()
            # load the image
            image = cv2.imread(path)
            image = cv2.resize(image, (28, 28))     
            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.blur(gray, (2,2))
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
            filename = "{}.png".format("./temp/{0}".format(filename))
            cv2.imwrite(filename, gray)
            '''
            # load the image as a PIL/Pillow image, apply OCR, and then delete the temporary file
            # 4 good pytesseract parameter
            text1 = pytesseract.image_to_string(Image.open(filename), config='--psm 1')
            list_string = text1.split()
            text6 = pytesseract.image_to_string(Image.open(filename), config='--psm 6')
            list_string.extend(text6.split())
            text4 = pytesseract.image_to_string(Image.open(filename), config='--psm 4')
            list_string.extend(text4.split())
            text3 = pytesseract.image_to_string(Image.open(filename), config='--psm 3')
            list_string.extend(text3.split())

            print("text1: ", text1)
            print("text6: ", text6)
            print("text4: ", text4)
            print("text3: ", text3)

            judge_value(list_string)
            # print(text)
            '''

def judge_value(list_string):
    dic_count = {}
    dic_count["100"] = list_string.count("100")
    dic_count["50"] = list_string.count("50")
    dic_count["20"] = list_string.count("20")
    dic_count["10"] = list_string.count("10")
    dic_count["5"] = list_string.count("5")
    dic_count["1"] = list_string.count("1")
    print("dic: ", dic_count)
    if dic_count["100"] == 0 and dic_count["50"] == 0 and dic_count["20"] == 0 and dic_count["10"] == 0 and dic_count["5"] == 0 and dic_count["1"] == 0:
        print("undefined")
        return
    face_value = max(dic_count.items(), key=operator.itemgetter(1))[0]
    print(">>>>>>>>>>", face_value, "banknote")
    
main()