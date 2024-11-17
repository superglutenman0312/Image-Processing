import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def histogram_equalize(img: np.ndarray, cdf: np.ndarray):
    height, width = img.shape
    equ_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            equ_img[i,j] = 255 * (cdf[img[i,j]] - min(cdf)) / (max(cdf) - min(cdf))
    return equ_img

def calc_hist(img: np.ndarray):
    height, width = img.shape
    hist = np.zeros(256, dtype = np.float32)
    for i in range(height):
        for j in range(width):
            hist[img[i,j]] += 1
    hist = hist / (height*width)
    
    cdf = np.zeros(256, dtype = np.float32)
    for i in range(hist.size):
        if i == 0:
            cdf[i] = hist[i]
        else:
            cdf[i] = hist[i] + cdf[i-1]
    return hist, cdf
    
def cut_and_combine_img(img: np.ndarray):
    height, width = img.shape
    cut_h, cut_w = 4, 4
    sub_img_h, sub_img_w = int(height/cut_h), int(width/cut_w)
    img_list = np.zeros((cut_h, cut_w, sub_img_h, sub_img_w), dtype = np.uint8)
    new_img_list = np.zeros((cut_h, cut_w, sub_img_h, sub_img_w), dtype = np.uint8)
    hist_list = np.zeros((cut_h, cut_w, 256), dtype = np.float32)
    cdf_list = np.zeros((cut_h, cut_w, 256), dtype = np.float32)
    new_hist_list = np.zeros((cut_h, cut_w, 256), dtype = np.float32)
    new_cdf_list = np.zeros((cut_h, cut_w, 256), dtype = np.float32)
    combine_img = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(cut_h):
        for j in range(cut_w):
            h_start, h_end = i*sub_img_h, (i+1)*sub_img_h
            w_start, w_end = j*sub_img_w, (j+1)*sub_img_w
            img_list[i,j] = img[h_start:h_end, w_start:w_end]
            hist_list[i,j], cdf_list[i,j] = calc_hist(img_list[i,j])
            new_img_list[i,j] = histogram_equalize(img_list[i,j], cdf_list[i,j])
            new_hist_list[i,j], new_cdf_list[i,j] = calc_hist(new_img_list[i,j])
            combine_img[h_start:h_end, w_start:w_end] = new_img_list[i,j]
    
    plt.figure('img_list')
    for i in range(cut_h):
        for j in range(cut_w):
            plt.subplot(cut_h, cut_w, i*cut_h + (j+1))
            plt.imshow(img_list[i,j], cmap='gray')
            
    plt.figure('img_list_histogram')
    for i in range(cut_h):
        for j in range(cut_w):
            plt.subplot(cut_h, cut_w, i*cut_h + (j+1))
            plt.plot(hist_list[i,j])
    
    plt.figure('img_list_cdf')
    for i in range(cut_h):
        for j in range(cut_w):
            plt.subplot(cut_h, cut_w, i*cut_h + (j+1))
            plt.plot(cdf_list[i,j])
            plt.ylim(0,1)
    
    plt.figure('new_img_list')
    for i in range(cut_h):
        for j in range(cut_w):
            plt.subplot(cut_h, cut_w, i*cut_h + (j+1))
            plt.imshow(new_img_list[i,j], cmap='gray')
            
    plt.figure('new_hist_list')
    for i in range(cut_h):
        for j in range(cut_w):
            plt.subplot(cut_h, cut_w, i*cut_h + (j+1))
            plt.plot(new_hist_list[i,j])
            
    plt.figure('new_cdf_list')
    for i in range(cut_h):
        for j in range(cut_w):
            plt.subplot(cut_h, cut_w, i*cut_h + (j+1))
            plt.plot(new_cdf_list[i,j])
            plt.ylim(0,1)
            
    return combine_img
    
if __name__ == '__main__':
    img = cv.imread('Lena.bmp', cv.IMREAD_GRAYSCALE)
    # img = cv.imread('Peppers.bmp', cv.IMREAD_GRAYSCALE)
    img_hist, img_cdf = calc_hist(img)
    img_equ = histogram_equalize(img, img_cdf)
    img_equ_hist, img_equ_cdf = calc_hist(img_equ)
    img_combine = cut_and_combine_img(img)
    img_combine_hist, img_combine_cdf = calc_hist(img_combine)
    
    plt.figure('result')
    plt.subplot(331)
    plt.imshow(img, cmap='gray')
    plt.subplot(334)
    plt.plot(img_hist)
    plt.subplot(337)
    plt.plot(img_cdf)
    plt.ylim(0,1)
    plt.subplot(332)
    plt.imshow(img_equ, cmap='gray')
    plt.subplot(335)
    plt.plot(img_equ_hist)
    plt.subplot(338)
    plt.plot(img_equ_cdf)
    plt.ylim(0,1)
    plt.subplot(333)
    plt.imshow(img_combine, cmap='gray')
    plt.subplot(336)
    plt.plot(img_combine_hist)
    plt.subplot(339)
    plt.plot(img_combine_cdf)
    plt.ylim(0,1)
    
    plt.show()