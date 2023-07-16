import cv2
import numpy as np
import scipy


def preProcess(img):
    """Grayscale + Blurs the img"""
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_gaussian = np.zeros((3,3))
    std = 0.572
    for index_x, i in enumerate(np.linspace(-1,1,3).astype(int)):
        for index_y, j in enumerate(np.linspace(-1,1,3).astype(int)):
            # print(i,j)
            kernel_gaussian[index_x,index_y] = np.exp(-1 * (i**2 + j**2) / (2 * std**2)) / (2*np.pi * std**2)

    blur = scipy.ndimage.convolve(gray_scale, kernel_gaussian, mode='reflect')

    filter_shapen = np.array([[0 ,0, 0],[0, .8, 0],[0, 0, 0]])
    sharpen = scipy.ndimage.convolve(blur, filter_shapen, mode='constant')
    threshold = cv2.adaptiveThreshold(sharpen, 255, 1, 1, 11, 2)
    return threshold

def findcorners(contour):
    """Find Corners of the Image"""
    corners = np.array([])
    best = -1
    for line in contour:
        current_area = cv2.contourArea(line)
        current_corners = cv2.approxPolyDP(line, 0.02* cv2.arcLength(line, closed = True), closed = True)
        if current_area > 60 and current_area > best and len(current_corners) == 4:
            best = current_area
            corners = current_corners
    return corners

def sort_points(corners):
    """Sort points to be compatible with warp perspective"""
    temp = corners.copy()
    # print(np.sum(temp, axis=1))
    
    top_left = np.argmin(np.sum(temp, axis=1))
    temp_2 = np.delete(temp, top_left, axis=0)
    bottom_right = np.argmax(np.sum(temp_2, axis=1))
    temp = np.delete(temp_2, bottom_right, axis=0)
    # print(temp)
    point_one = temp[0]
    point_two = temp[1]
    
    top_right = None
    bottom_left = None
    if point_one[0] > point_two[0] and point_one[1] < point_two[1]:
        top_right = 0 #point 1
        bottom_left = 1 #point 2
    else:
        bottom_left = 0 #point 1
        top_right = 1 #point 2
    reordered = np.array([[corners[top_left]],
                          [temp[top_right]],
                          [temp[bottom_left]],
                          [temp_2[bottom_right]]])

    return reordered


def split_img(img):
    rows = np.vsplit(img,9)
    
    grid = [[0 for i in range(9)] for i in range(9)]
    for row, r in enumerate(rows):
        cols= np.hsplit(r,9)
        for col, box in enumerate(cols):
            grid[row][col] = box
    return grid