import numpy as np
import sys
import cv2 as cv
import os

def show_wait(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def remove_table_lines(imgPath, remove_horizontal=True, remove_vertical=True):

    if not os.path.exists(imgPath):
        print("imageg not exists!")
        return -1
    
    # Load the image
    src = cv.imread(imgPath, cv.IMREAD_COLOR)
    if src is None:
        print ('Error opening image: ' + argv[0])
        return -1

    # gray and bin
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src
    show_wait("gray", gray)
    gray_bw = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray_bw, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 15, -2)
    show_wait("binary", bw)

    # initialize
    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    if  remove_horizontal:
        # 1.get bitwise
        cols = horizontal.shape[1]
        horizontal_size = 20  # this parameter is about the metric of font, just try to change it when final effect is not good
        horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
        horizontal = cv.erode(horizontal, horizontalStructure)
        horizontal = cv.dilate(horizontal, horizontalStructure)
        horizontal = cv.bitwise_not(horizontal)
        show_wait("horizontal_bitwise", horizontal)
        # 2.extract edges
        horizontal_edges = cv.adaptiveThreshold(horizontal, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 3, -2)
        kernel = np.ones((2, 2), np.uint8)
        horizontal_edges = cv.dilate(horizontal_edges, kernel)
        show_wait("dilate_edges", horizontal_edges)
        (rows, cols) = np.where(horizontal_edges > 200)
        gray[rows, cols] = 255
        # Show image without horizontal_edges
        show_wait("removed horizontal_edges", gray)

    if  remove_vertical:  
        # 1.get bitwise
        rows = vertical.shape[0]
        vertical_size =  20  # this parameter is about the metric of font, just try to change it when final effect is not good
        verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
        vertical = cv.erode(vertical, verticalStructure)
        vertical = cv.dilate(vertical, verticalStructure)
        vertical = cv.bitwise_not(vertical)
        show_wait("vertical_bitwise", vertical)
        # 2.extract edges
        vertical_edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                    cv.THRESH_BINARY, 3, -2)
        kernel = np.ones((2, 2), np.uint8)
        vertical_edges = cv.dilate(vertical_edges, kernel)
        show_wait("dilate_vertical", vertical_edges)
        (rows, cols) = np.where(vertical_edges > 200)
        gray[rows, cols] = 255
        # Show image without vertical_edges
        show_wait("removed vertical_edges", gray)

    return 0

if __name__ == "__main__":
    image_Path = r"C:\Users\zhak\GFTE\output\img\0801.2890v1.1.png"
    remove_table_lines(image_Path, True, True)