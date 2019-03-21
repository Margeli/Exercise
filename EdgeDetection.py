import numpy as np
import cv2


def convolve(dest, src, i, j, kernel):
    krows, kcols = kernel.shape
    srctmp = src[i:i + krows, j:j + kcols]
    dest[i, j] = (srctmp * kernel[:, :, np.newaxis]).sum(axis=(0, 1))


def mixpixels(dest, src, i, j, kernel):
    krows, kcols = kernel.shape
    final = src[i][j]
    srctmp = src[i:i + krows, j:j + kcols]

    for r in range(0, krows):
        for c in range(0,kcols):
            final += srctmp[r][c] * (kernel[r][c])

    dest[i][j] = final


def applyGaussFilter(dest):

    rows, cols, channels = dest.shape
    # Kernel size / radius
    ksize = 5
    kradi = ksize // 2

    # Create the kernel manually
    kernel = np.array([
        [2, 4, 5, 4, 2],
        [4, 9, 12, 9, 4],
        [5, 12, 15, 12, 5],
        [4, 9, 12, 9, 4],
        [2, 4, 5, 4, 2]
    ])
    # Creating the kernel with opencv
    # kradi = ksize / 2
    # sigma = np.float64(kradi) / 2
    # kernel = cv2.getGaussianKernel(ksize, sigma)
    # kernel = np.repeat(kernel, ksize, axis=1)
    # kernel = kernel * kernel.transpose()
    # kernel = kernel / kernel.sum()

    # Create a copy with black padding
    imgpadding = np.zeros((rows + 2 * kradi, cols + 2 * kradi, channels))
    imgpadding[kradi:-kradi, kradi:-kradi] = dest

    # Convolution
    src = np.zeros(dest.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            convolve(src, imgpadding, i, j, kernel)
    src /= kernel.sum()
    return src

def applySobelFilter(dest):

    rows, cols, channels = dest.shape
    # Kernel size / radius
    ksize = 3
    kradi = ksize // 2

    # Create the kernel manually
    h_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    v_kernel = np.array([
        [-1,2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    # Create a copy with black padding
    imgpadding = np.zeros((rows + 2 * kradi, cols + 2 * kradi, channels))
    imgpadding[kradi:-kradi, kradi:-kradi] = dest

    # Convolution
    h_src = np.zeros(dest.shape)
    for i in range(0, rows):
        for j in range(0, cols):
           convolve(h_src, imgpadding, i,j,h_kernel)

    # h_src /= h_kernel.sum() WHY???
    cv2.imshow("Sobel_h_Filtered", np.uint8(h_src))


    v_src = np.zeros(dest.shape)
    for l in range(0, rows):
        for m in range(0, cols):
            convolve(v_src, imgpadding, l, m, v_kernel)

    # v_src /= v_kernel.sum()
    cv2.imshow("Sobel_v_Filtered", np.uint8(v_src))

    src = np.zeros(dest.shape)
    for i in range(0, rows):
        for j in range(0, cols):
           src[i][j] = np.sqrt(((h_src[i][j])**2)+ ((v_src[i][j])**2))

    cv2.imshow("Sobel_Filtered", np.uint8(src))
    return src

def run():
    # Load an image
    img = cv2.imread("image2.png", cv2.IMREAD_ANYCOLOR)

    img2 = applyGaussFilter(img)
    # cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY);


    img3 = applySobelFilter(img2)
    # Using cv2
    # filtered = cv2.GaussianBlur(img, (5,5), 2.0)

    # Show the image

    # cv2.imshow("Original", img)
    # cv2.imshow("Gauss_Filtered", np.uint8(img2))
    sobel = cv2.Sobel(img2,cv2.CV_64F,1,0 );
    sobely = cv2.Sobel(img2,cv2.CV_64F,0,1 );
    cv2.imshow("Sobel x correct", np.uint8(sobel))
    cv2.imshow("Sobel y correct", np.uint8(sobely))
    cv2.waitKey(0)


if __name__ == "__main__":
    run()
