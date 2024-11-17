import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

S_factor = 1.5; I_factor = 4; L_factor = 4

def Lab_enhance(img: np.ndarray):
    enhanced_img = img.copy()
    enhanced_img[:, :, 0] *= L_factor
    # enhanced_img[:, :, 0] = np.clip(enhanced_img[:, :, 0], 0, 100)
    return enhanced_img 

def HSI_enhance(hsi_img: np.ndarray):
    enhanced_hsi_img = hsi_img.copy()
    enhanced_hsi_img[:, :, 1] *= S_factor
    enhanced_hsi_img[:, :, 1] = np.clip(enhanced_hsi_img[:, :, 1], 0, 1)
    enhanced_hsi_img[:, :, 2] *= I_factor
    enhanced_hsi_img[:, :, 2] = np.clip(enhanced_hsi_img[:, :, 2], 0, 1)
    return enhanced_hsi_img

def RGB2HSI(img: np.ndarray):
    height, width, channel = img.shape
    hsi_img = np.zeros((height, width, 3), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            r, g, b = img[i, j] / 255.0
            num = 0.5 * ((r - g) + (r - b))
            den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
            theta = np.where(den == 0, 0, np.arccos(num / den))
            H = np.where(b <= g, theta, 2*np.pi-theta)
            S = 1-(3/(r+g+b))*min(r,g,b)
            I = (r + g + b) / 3.0
            hsi_img[i, j, 0] = H
            hsi_img[i, j, 1] = S
            hsi_img[i, j, 2] = I
    return hsi_img

def HSI2RGB(hsi_img: np.ndarray):
    height, width, channel = hsi_img.shape
    rgb_img = np.zeros((height, width, 3), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            H, S, I = hsi_img[i, j]
            if 0 <= H < (2/3)*np.pi:
                B = I * (1 - S)
                R = I * (1 + (S * np.cos(H) / np.cos((1/3)*np.pi - H)))
                G = 3 * I - (R + B)
            elif (2/3)*np.pi <= H < (4/3)*np.pi:
                H -= (2/3) * np.pi
                R = I * (1 - S)
                G = I * (1 + (S * np.cos(H) / np.cos((1/3)*np.pi - H)))
                B = 3 * I - (R + G)
            elif (4/3)*np.pi <= H < 2*np.pi:
                H -= (4/3)*np.pi
                G = I * (1 - S)
                B = I * (1 + (S * np.cos(H) / np.cos((1/3)*np.pi - H)))
                R = 3 * I - (G + B)
            rgb_img[i, j, 0] = R * 255
            rgb_img[i, j, 1] = G * 255
            rgb_img[i, j, 2] = B * 255
            rgb_img[i, j, 0] = np.clip(rgb_img[i, j, 0], 0, 255)
            rgb_img[i, j, 1] = np.clip(rgb_img[i, j, 1], 0, 255)
            rgb_img[i, j, 2] = np.clip(rgb_img[i, j, 2], 0, 255)
            
    return rgb_img.astype(np.uint8)

if __name__ == '__main__':
    img = cv.imread('aloe.jpg') # S_factor = 1.5; I_factor = 4; L_factor = 4
    # img = cv.imread('church.jpg') # S_factor = 1.5; I_factor = 2.5; L_factor = 2.5
    # img = cv.imread('house.jpg') # S_factor = 1.3; I_factor = 0.8; L_factor = 0.9
    # img = cv.imread('kitchen.jpg') # S_factor = 0.7; I_factor = 0.9; L_factor = 0.8
    
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    hsi_img = RGB2HSI(img.astype(np.float64))
    enhanced_hsi_img = HSI_enhance(hsi_img)
    hsi_enhanced_img = HSI2RGB(enhanced_hsi_img)
    
    Lab_img = cv.cvtColor(img, cv.COLOR_RGB2LAB).astype(np.float64)
    enhanced_Lab_img = Lab_enhance(Lab_img)
    enhanced_Lab_img = np.clip(enhanced_Lab_img, 0, 255)
    enhanced_Lab_img = enhanced_Lab_img.astype(np.uint8)
    Lab_enhanced_img = cv.cvtColor(enhanced_Lab_img, cv.COLOR_LAB2RGB)
    
    plt.figure('Result')
    plt.subplot(131)
    plt.title('Original image')
    plt.imshow(img.astype(np.uint8))
    
    plt.subplot(132)
    plt.title(f'HSI enhanced img\n S factor = {S_factor}, I factor = {I_factor}')
    plt.imshow(hsi_enhanced_img.astype(np.uint8))
    
    plt.subplot(133)
    plt.title(f'Lab enhanced img\n L factor = {L_factor}')
    plt.imshow(Lab_enhanced_img.astype(np.uint8))
    
    plt.show()
