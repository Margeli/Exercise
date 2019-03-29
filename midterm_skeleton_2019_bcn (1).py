########################################################################
#                         TYPE YOUR NAME HERE
########################################################################

import cv2
import numpy as np

def convolve(dest, src, i, j, kernel):
    krows, kcols = kernel.shape
    final = src[i][j]
    srctmp = src[i:i + krows, j:j + kcols]

    for r in range(0, krows):
        for c in range(0,kcols):
            final += srctmp[r][c] * (kernel[r][c])

    dest[i][j] = final


def exercise1():

    img = np.float64(cv2.imread("noise.jpg", cv2.IMREAD_COLOR))

    rows, cols, comp = img.shape


    ksize = 11
    kradi = ksize//2

    k1 = np.array([1,10,45,120,210,252,210,120,45,10,1])

    kernel = np.zeros((ksize,ksize))
    kernel[kradi,:] = k1

    padding = np.zeros((rows+2*kradi, cols + 2*kradi, 3))
    padding[kradi:-kradi,kradi:-kradi ] = img

    src = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            convolve(src, padding, i, j, kernel)
    src /= kernel.sum()

    kernel2 = np.zeros((ksize, ksize))
    kernel2[:, kradi] = k1

    padding2 = np.zeros((rows+2*kradi, cols + 2*kradi, 3))
    padding2[kradi:-kradi,kradi:-kradi ] = src

    src2 = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            convolve(src2, padding2, i, j, kernel2)
    src2 /= kernel.sum()

    cv2.imshow("Original", np.uint8(img))
    cv2.imshow("Blurred", np.uint8(src))
    cv2.imshow("Blurred2", np.uint8(src2))
    cv2.waitKey(0)

def CheckPixel(img, x, y, kernel):
    rows, cols = img.shape
    krows, kcols = kernel.shape
    kradi= krows//2

    tmpimg = np.zeros((krows, kcols))
    tmpimg = img[x-kradi:x+kradi+1, y-kradi:y+kradi+1]

    for i in range(0,krows):
        for j in range(0, kcols):
            if tmpimg[i, j] == 0:
                return 0

    return 255


def exercise2():
    img = cv2.imread("morphology.png", cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape
    ksize = 3
    kradi= ksize//2
    kernel = np.zeros((ksize, ksize))

    # TODO: Write your code here
    # padding = np.zeros((rows+2*kradi, cols+2*kradi))
    # padding[ksize:-ksize, ksize:-ksize] = img

    src = np.ones((rows, cols))
    for x in range(ksize,rows-ksize):
        for y in range(ksize,cols-ksize):
            src[x,y] = CheckPixel(img,x, y, kernel )



    cv2.imshow("Input", img)
    cv2.imshow("Output", np.uint8(src))
    cv2.waitKey(0)


if __name__ == '__main__':

    # Uncomment to execute exercise 1
     #exercise1()

    # Uncomment to execute exercise 2
    exercise2()


