import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

laplacian_filter = np.array([[ 0, 1, 0],
                             [ 1,-4, 1],
                             [ 0, 1, 0]])

def laplacian_calc_v1(img: np.ndarray):
    height, width = img.shape
    filter_img = np.zeros((height, width), dtype = np.float32)
    for i in range(1,height-1):
        for j in range(1,width-1):
            filter_img[i,j] = ((img[i-1,j] + img[i+1,j] + img[i,j-1] + img[i,j+1]) - 4*img[i,j])
    result_img = filter_img[1:-1, 1:-1]
    result_img = np.clip(result_img, 0, 255) 
    return result_img

def laplacian_calc_v2(img: np.ndarray):
    global laplacian_filter
    filter_img = cv.filter2D(img, -1, laplacian_filter)
    return filter_img 

def highboost_calc(img: np.ndarray):
    A = 1.7
    blur_img = cv.GaussianBlur(img, (7,7), 0, 0)
    img2 = img.astype(np.float32)
    sharpened_img = img2 - blur_img
    hb_img_v1 = (A-1) * img2 + sharpened_img
    hb_img_v2 = A * img - laplacian_calc_v2(img)    
    return sharpened_img, hb_img_v1, hb_img_v2

if __name__ == '__main__':
    img = cv.imread('blurry_moon.tif')
    # img = cv.imread('skeleton_orig.bmp')
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # laplacian v1
    padded_img = cv.copyMakeBorder(gray_img, 1,1,1,1, cv.BORDER_DEFAULT)
    padded_img = padded_img.astype(np.float32)
    laplacian_filter_img = laplacian_calc_v1(padded_img)
    laplacian_img = gray_img + laplacian_filter_img
    laplacian_img = laplacian_img.astype(np.uint8)
    
    # laplacian v2
    laplacian_filter_img = laplacian_calc_v2(gray_img)
    laplacian_img = gray_img + laplacian_filter_img
    
    # high-boosted    
    hb_filter_img, hb_img_v1, hb_img_v2 = highboost_calc(gray_img)
    
    # draw diagram
    plt.figure('Expermient result')
    plt.subplot(231)
    plt.title('figure 1')
    plt.imshow(gray_img, cmap='gray')
    plt.subplot(232)
    plt.title('figure 2')
    plt.imshow(laplacian_filter_img, cmap='gray')
    plt.subplot(233)
    plt.title('figure 3')
    plt.imshow(laplacian_img, cmap='gray')
    plt.subplot(234)
    plt.title('figure 4')
    plt.imshow(hb_filter_img, cmap='gray')
    plt.subplot(235)
    plt.title('figure 5')
    plt.imshow(hb_img_v1, cmap='gray')
    plt.subplot(236)
    plt.title('figure 6')
    plt.imshow(hb_img_v2, cmap='gray')

    plt.figure('compare')
    plt.subplot(131)
    plt.title('my function\'s sharpened image (float32)')
    padded_img = cv.copyMakeBorder(gray_img, 1,1,1,1, cv.BORDER_DEFAULT)
    padded_img = padded_img.astype(np.float32)
    laplacian_filter_img = laplacian_calc_v1(padded_img)
    plt.imshow(laplacian_filter_img,cmap='gray')
    plt.subplot(132)
    plt.title('my function\'s sharpened image (uint8)')
    laplacian_filter_img = laplacian_filter_img.astype(np.uint8)
    plt.imshow(laplacian_filter_img,cmap='gray')
    plt.subplot(133)
    plt.title('filter2D\'s sharpened image (uint8)')
    laplacian_filter_img = laplacian_calc_v2(gray_img)
    plt.imshow(laplacian_filter_img,cmap='gray')


    plt.show()
