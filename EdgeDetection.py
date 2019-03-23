import numpy as np
import cv2




def applyFilter(img, filter):
    newimg = np.float64(img.copy())
    rows, columns = img.shape
    f_rows, f_columns = filter.shape
    f_rows_half = np.uint8(f_rows / 2)
    f_columns_half = np.uint8(f_columns / 2)
    for x in range(0, rows):
        for y in range(0, columns):
            submat = img[max(0, x-f_rows_half):min(rows, x+f_rows_half+1), max(0, y-f_columns_half):min(columns, y+f_columns_half+1)]
            f_submat = filter[max(f_rows_half-x, 0):f_rows-max(0, x+f_rows_half-rows+1), max(f_columns_half-y, 0):f_columns-max(0, y+f_columns_half-columns+1)]
            newimg[x, y] = np.sum(submat*f_submat)
    return newimg

def mixpixels(dest, src, i, j, kernel):
    krows, kcols = kernel.shape
    final = src[i][j]
    srctmp = src[i:i + krows, j:j + kcols]

    for r in range(0, krows):
        for c in range(0,kcols):
            final += srctmp[r][c] * (kernel[r][c])

    dest[i][j] = final


def applyGaussFilter(dest):

    rows, cols = dest.shape
    # Kernel size / radius
    ksize = 5
    kradi = ksize // 2

    # Create the kernel manually
    kernel = np.array(([1, 4, 7, 4,1],
             [4,16,26,16,4],
             [7,26,41,26,7],
             [4,16,26,16,4],
             [1, 4, 7, 4,1]))
    # Creating the kernel with opencv
    # kradi = ksize / 2
    # sigma = np.float64(kradi) / 2
    # kernel = cv2.getGaussianKernel(ksize, sigma)
    # kernel = np.repeat(kernel, ksize, axis=1)
    # kernel = kernel * kernel.transpose()
    # kernel = kernel / kernel.sum()

    # Create a copy with black padding
    imgpadding = np.zeros((rows + 2 * kradi, cols + 2 * kradi))
    imgpadding[kradi:-kradi, kradi:-kradi] = dest

    # Convolution
    src = np.zeros(dest.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            mixpixels(src, imgpadding, i, j, kernel)
    src /= kernel.sum()
    return src

def getproxAngle(angle, min, max): # 120, 90, 135
    if abs(angle-min)> abs(angle-max):
        return max
    else:
        return min

def capAngles(angle):
    if(angle<0):
        angle+=180
    if angle > 0 and angle<=45:
        return getproxAngle(angle, 0, 45)
    if angle > 45 and angle<=90:
        return getproxAngle(angle, 45, 90)
    if angle > 90 and angle<=135:
        return getproxAngle(angle, 90, 135)
    if angle > 135 and angle<=180:
        ang = getproxAngle(angle, 135, 180)
        if(ang ==180):
            return 0
        else:
            return ang #135

def applySobelFilter(dest):

    rows, cols = dest.shape
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
        [-1,-2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    # Create a copy with black padding
    imgpadding = np.zeros((rows + 2 * kradi, cols + 2 * kradi))
    imgpadding[kradi:-kradi, kradi:-kradi] = dest

    # Convolution

    h_src = applyFilter(dest, h_kernel)
    # for i in range(0, rows):
    #     #     for j in range(0, cols):
    #     #        convolve(h_src, imgpadding, i,j,h_kernel)


    # cv2.imshow("My_Sobel_x_Filtered", np.uint8(h_src))


    v_src = applyFilter(dest, v_kernel)
    #
    # cv2.imshow("My_Sobel_y_Filtered", np.uint8(v_src))

    src = np.zeros(dest.shape)
    for i in range(0, rows):
        for j in range(0, cols):
           src[i][j] = np.sqrt(((h_src[i][j])**2)+ ((v_src[i][j])**2))

    cv2.imshow("My_Sobel_Filtered", np.uint8(src))

    angles = np.arctan2(h_src, v_src) #0, 45, 90, 135
    for x in range(rows):
        for y in range(cols):
            angles[x][y] = capAngles(np.rad2deg(angles[x][y]))

    cv2.imshow("My_Sobel_Filtered_Angles", np.uint8(angles))

    return src, angles

def applyMFilter(mag, ang):
    rows, cols = mag.shape
    img =  np.zeros(mag.shape)
    for x in range(1,rows-1):
        for y in range(1,cols-1):
            if ang[x][y]== 0:
                if mag[x][y] >= mag[x+1][y] and mag[x][y] >= mag[x-1][y]:
                    img[x][y] = mag[x][y]
            if ang[x][y]== 45:
                if mag[x][y] >= mag[x][y+1] and mag[x][y] >= mag[x][y-1]:
                    img[x][y] = mag[x][y]
            if ang[x][y]== 90:
                if mag[x][y] >= mag[x-1][y+1] and mag[x][y] >= mag[x+1][y-1]:
                    img[x][y] = mag[x][y]
            if ang[x][y] == 135:
                if mag[x][y] >= mag[x - 1][y-1] and mag[x][y] >= mag[x + 1][y+1]:
                    img[x][y] = mag[x][y]
    return img

def applyHysteresisThresholdingFilter(img, min, max):
    nwimg = np.zeros(img.shape)
    rows, cols = img.shape
    for x in range(rows):
        for y in range(cols):
            if img[x][y] >= max: #edge
                nwimg[x][y] = 255
            if img[x][y] <= min: #not edge
                nwimg[x][y] = 0
            if img[x][y]<max and img[x][y]>min:
                nwimg[x][y] = 0.5 #posible
    for x in range(rows):
        for y in range(cols):
            if nwimg[x][y] == 0.5:
                for i in range(-1,2):
                    for j in range(-1, 2):
                        if nwimg[x+i][y+j] == 255 :
                            nwimg[x][y] = 255

    return nwimg

def run():
    # Load an image
    img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

    # Using cv2
    # filtered = cv2.GaussianBlur(img, (5,5), 2.0)
    blurr = applyGaussFilter(img)
    g_magnitudes,g_angles = applySobelFilter(blurr)

    m_img = applyMFilter(g_magnitudes, g_angles)

    canny = applyHysteresisThresholdingFilter(m_img, 20, 50)

    cv2.imshow("M", np.uint8(m_img))
    cv2.imshow("Canny", np.uint8(canny))
    # Show the image

    # cv2.imshow("Original", img)
    # cv2.imshow("Gauss_Filtered", np.uint8(img2))
    # sobel = cv2.Sobel(img2,cv2.CV_64F,1,0 );
    # # sobely = cv2.Sobel(img2,cv2.CV_64F,0,1 );
    # cv2.imshow("Sobel x correct", np.uint8(sobel))
    # cv2.imshow("Sobel y correct", np.uint8(sobely))

    cv2.waitKey(0)


if __name__ == "__main__":
    run()
