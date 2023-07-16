import cv2
import numpy as np
import scipy
from helper import preProcess, findcorners, sort_points, split_img
from solver import solve
from tensorflow import keras # Library for neural networks
from tensorflow.keras.models import load_model
# Import the saved model
# model = keras.models.load_model('mnist_nnet.h5')
model = load_model('mnist_nnet.h5')

pathImage = "boards/Sudoku_1.png"
heightimg = 450
widthimage = 450

# Step 1: Process Image 
load = cv2.imread(pathImage)
img = cv2.resize(load, (widthimage, heightimg))

img_blank = np.zeros((heightimg, widthimage, 3), np.int8)
processed_img = preProcess(img)

cv2.imwrite("test_step1.png", img= processed_img)

#Step 2: Contouring Image to find the border:
contours = img.copy() # Debugging
img_contours, temp = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contours, img_contours, -1, (0, 255, 0), 3) # Debugging
cv2.imwrite("test_step2.png", img= contours) # Debugging

#Step 3: Find corners of image use warp perspective to fit the corners into new image
corners = findcorners(contour=img_contours)
# print(corners)
if corners.size == 0:
    print("Board cannot be found")
    exit(1)
    
corners = corners.reshape((4,2))
corners = sort_points(corners)

coutours_two = img.copy() # Debugging
cv2.drawContours(coutours_two, corners, -1, (0, 255, 0), 25) # Debugging
cv2.imwrite("test_step3.png", img= coutours_two) # Can delete
pts1 = np.float32(corners) 
pts2 = np.float32([[0, 0],[widthimage, 0], [0, heightimg],[widthimage, heightimg]]) 
H = cv2.getPerspectiveTransform(pts1, pts2) 
imgWarped = cv2.warpPerspective(img, H, (widthimage, heightimg))

cv2.imwrite("warped.png", img= imgWarped) # Can delete


#Step 4: Isolate Each box:
imgWarpColored = cv2.cvtColor(imgWarped,cv2.COLOR_BGR2GRAY)
Grid = split_img(imgWarpColored)

Board = np.zeros((9, 9))
cv2.imwrite("tester.png", Grid[0][0])
box = np.asarray(Grid[0][0])

#Step 5: Classifify each box
for row in range(0, 9):
    for col in range(0, 9):
        cv2.imwrite(f"images/tester{row}{col}.png", Grid[row][col])
        box = np.asarray
        box = np.asarray(Grid[row][col])
        box = box[6:box.shape[0] - 6, 6:box.shape[1] -6]
        # Invert colors and apply threshold to make the image similar to MNIST dataset
        _, box = cv2.threshold(255 - box, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        box = cv2.resize(box, (28, 28))
        box = box / 255
        box = box.reshape(1, 28, 28, 1)

        prediction = model.predict(box)
        classIndex = np.argmax(prediction)
        threshold = 0.5
        total_pixel_intensity = np.sum(box)
        empty_cell_threshold = 40  # Adjust this value to fit your needs
        print(total_pixel_intensity)
        
        if total_pixel_intensity < empty_cell_threshold:
            classIndex = 0
            Board[row][col] = classIndex
        elif classIndex == 0 and total_pixel_intensity > empty_cell_threshold:
            classIndex = 6
            Board[row][col] = classIndex
        else:
            Board[row][col] = classIndex

        print(classIndex)

        
        
print("Detected Sudoku board:")
print(Board)

print("Solved Sudoku board:")
solve(board=Board)

