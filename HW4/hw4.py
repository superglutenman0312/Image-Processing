import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

sobel_op_x = np.array([[-1,-2,-1],
                       [ 0, 0, 0],
                       [ 1, 2, 1]], dtype = np.float32)

sobel_op_y = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype = np.float32)

def sobel_filter_op(img: np.ndarray):
    height, width = img.shape
    img = img.astype(np.float32)
    filter_img_x = np.zeros((height, width), dtype = np.float32)
    filter_img_y = np.zeros((height, width), dtype = np.float32)
    for i in range(1,height-1):
        for j in range(1,width-1):
            filter_img_x[i,j] = ( - (img[i-1,j-1] + 2*img[i-1,j] + img[i-1,j+1]) + (img[i+1,j-1] + 2*img[i+1,j] + img[i+1,j+1]) ) # sobel_op_x filter's mask
            filter_img_y[i,j] = ( - (img[i-1,j-1] + 2*img[i,j-1] + img[i+1,j-1]) + (img[i-1,j+1] + 2*img[i,j+1] + img[i+1,j+1]) ) # sobel_op_y filter's mask       
    result_img_x = filter_img_x[1:-1, 1:-1]
    result_img_y = filter_img_y[1:-1, 1:-1]
    result_img_x = np.clip(result_img_x, 0, 255)
    result_img_y = np.clip(result_img_y, 0, 255)    
    return result_img_x.astype(np.uint8), result_img_y.astype(np.uint8)

def padding(img: np.ndarray):
    height, width = img.shape
    padded_img = np.zeros((height+2, width+2), dtype = np.uint8)
    # 四邊
    padded_img[1:-1, 1:-1] = img
    padded_img[1:-1,0] = img[:,0]
    padded_img[1:-1,-1] = img[:,-1]
    padded_img[0,1:-1] = img[0,:]
    padded_img[-1,1:-1] = img[-1,:]
    # 四角
    padded_img[0,0] = img[0,0]
    padded_img[0,-1] = img[0,-1]
    padded_img[-1,0] = img[-1,0]
    padded_img[-1,-1] = img[-1,-1]
    return padded_img

if __name__ == '__main__':
    img = cv.imread('87.png')
    # img = cv.imread('baboon.png')
    # img = cv.imread('pool.png')
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    padded_img = padding(gray_img)
    
    # use my function to implement convolution calculation
    filter_img_x, filter_img_y = sobel_filter_op(padded_img)
    combined_img = np.sqrt(filter_img_x * filter_img_x + filter_img_y * filter_img_y)
    combined_img = np.clip(combined_img, 0, 255)
    combined_img = combined_img.astype(np.uint8)

    plt.figure('result')
    plt.subplot(221)
    plt.title('origin gray image')
    plt.imshow(gray_img, cmap = 'gray')
    plt.subplot(222)
    plt.title('|Gx| component in the x-direction')
    plt.imshow(filter_img_x, cmap = 'gray')
    plt.subplot(223)
    plt.title('|Gy| component in the y-direction')
    plt.imshow(filter_img_y, cmap = 'gray')
    plt.subplot(224)
    plt.title('Gradient image, |Gx| + |Gy|')
    plt.imshow(combined_img, cmap = 'gray')

    # use filter2D to implement convolution calculation
    filter_img_x2 = cv.filter2D(gray_img, -1, sobel_op_x)
    filter_img_y2 = cv.filter2D(gray_img, -1, sobel_op_y)
    combined_img2 = filter_img_x2 + filter_img_y2
    combined_img2 = np.clip(combined_img2, 0, 255)
    combined_img2 = combined_img2.astype(np.uint8)

    plt.figure('result with cv2.filter2D()')
    plt.subplot(221)
    plt.title('origin gray image')
    plt.imshow(gray_img, cmap = 'gray')
    plt.subplot(222)
    plt.title('|Gx| component in the x-direction')
    plt.imshow(filter_img_x2, cmap = 'gray')
    plt.subplot(223)
    plt.title('|Gy| component in the y-direction')
    plt.imshow(filter_img_y2, cmap = 'gray')
    plt.subplot(224)
    plt.title('Gradient image, |Gx| + |Gy|')
    plt.imshow(combined_img2, cmap = 'gray')

    plt.show()