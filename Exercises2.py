import numpy as n
import cv2



def Ex1():
    img = cv2.imread("image.png", cv2.IMREAD_ANYCOLOR)

    print('The shape of the image is {}'.format(img.shape))
    print('The num of dimensions of the image is {}'.format(len(img.shape)))
    print('The internal type of the image is {}'.format(img.dtype))


def Ex2():
    img = cv2.imread("image.png", cv2.IMREAD_ANYCOLOR)
    img = n.float64(img)/255


def Ex3():
    img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
    threshold = 125
    img = n.float64(img>threshold)
    cv2.imshow("image", n.uint8(img * 255))
    cv2.waitKey(0)

def ex4():
    print('Ex4 - Open an image and apply a vignetting effect on its borders')
    img = cv2.imread('image.jpg', cv2.IMREAD_ANYCOLOR)
    img = n.float64(img)
    height, width,_ = img.shape
    centeri = height / 2
    centerj = width / 2
    radius = n.sqrt(width * width + height * height) / 2.0
    for i in range(0, height):
        for j in range(0, width):
            dist = n.sqrt((centeri - i) ** 2 + (centerj - j) ** 2)
            vignetting_factor = 1.0 - (dist / radius) ** 2
            img[i, j] *= vignetting_factor
    cv2.imshow('Image', n.uint8(img))
    cv2.waitKey(0)

def Ex5():
    img = cv2.imread("image.png", cv2.IMREAD_ANYCOLOR)
    img = n.float64(255-img)
    cv2.imshow('Image', n.uint8(img))
    cv2.waitKey(0)

def ex6():
    print("Ex6 - Open an image and tint it 50% with blue")
    img = cv2.imread('image.jpg', cv2.IMREAD_ANYCOLOR)
    img = n.float64(img)
    blue = n.array([255.0, 0.0, 0.0]).reshape((1, 1, 3))
    img = 0.5 * img + 0.5 * blue
    cv2.imshow('Image', n.uint8(img))
    cv2.waitKey(0)

def Ex7():
    img = cv2.imread('image.jpg', cv2.IMREAD_ANYCOLOR)
    img = n.float64(img)
    min = 50
    max = 200
    img= (img-min)/ (max-min)
    img = n.clip(img,0,1)
    cv2.imshow('Image', n.uint8(img*255))
    cv2.waitKey(0)

#
# if __name__ == "__main__":
#     Ex1()
#     # Ex2()
#     # Ex3()
#     # ex4()
#     # Ex5()
#     # ex6()
#     Ex7()
