import numpy as n
import cv2



def Ex1():
    img = cv2.imread("image.png")

    imgInfo = n.dtype
    print(img.size())
    cv2.imshow("Image", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    Ex1()
