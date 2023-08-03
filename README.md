# Sudoku Solver
## Sai Vaka, Aniket Nedunuri, Ankith Palakodati

## Description
This project aims to create an automated Sudoku solver that can take an image of an unsolved Sudoku puzzle, recognize the digits, and efficiently provide the solution. The solver utilizes computer vision techniques to preprocess the image, extract individual cells, and apply a neural network to classify the digits. The puzzle is then solved using a backtracking algorithm. This program removes the process for a user to enter manually enter numbers into an array and can simply pass a picture of the board, resulting in less work done for the user.

## Approach
The project follows a step-by-step process:
1. Image Processing: Apply filters and thresholding techniques to enhance contrast between digits, borders, and the background.
2. Border Identification: Use contouring techniques to detect the puzzle's border.
3. Corner Detection: Find the corners of the puzzle using the approxPolyDP algorithm.
4. Board Splitting: Isolate the Sudoku board using a homography transformation and split the image into 81 equal square cells.
5. Digit Classification: Pass each cell of the board through a neural network trained on the MNIST dataset to identify the digits.
6. Puzzle Solving: Utilize a backtracking algorithm to solve the Sudoku puzzle based on the classified digits.

## Experiments
For training and testing the robustness of the digit recognition component of the neural network, a diverse dataset of digit images were collected from the MNIST dataset. The success of the project was measured through metrics such as accuracy of digit recognition on isolated cells, success rate of puzzle solving compared to random chance, and correctness of the solved puzzles.

## Results
Provided is both the input and output of the program. The input is an image provided by the user and the output is printed out into the terminal.
<p align="center">
   <img src="https://github.com/saivaka/OpenCV-Sudoku-Solver/blob/main/Results/Before.png" alt="Input" width="380" height="380"/>
   <img src="https://github.com/saivaka/OpenCV-Sudoku-Solver/blob/main/Results/After.png" alt="Output" width="380" height="380"/>
</p>

## Installation

What are the steps required to install your project? Provide a step-by-step description of how to get the development environment running.

## Usage
Run the Preprocess.py script, providing the path to an image of an unsolved Sudoku puzzle. The script will preprocess the image with filters and contouring, classify the digits, solve the puzzle, and display the solved puzzle on the terminal. Results. The project demonstrates significant advancements in computer vision techniques, aided by sophisticated algorithms, large training datasets, and improved computational power. The combination of image processing, machine learning, and backtracking techniques enables efficient and accurate Sudoku puzzle solving.

### Future Enhancements:
-Make the program accept an image as an arg input instead of manually entering path in the source code   
-Add a requirements.txt file of any needed installations to run the program    
-Improve the robustness of the OpenCV filtering and digit recognition neural network model to handle variations in lighting, angles, and image quality. We found that some sudoku boards didn't work if they were distorted or angled in a certain manner. However, by optimizing our homographies, filtering, and neural network model, we can improve our program to work on almost any sudoku boards, including handwritten boards.      
-Potentially print the digits back onto the board image provide by the user    

Please refer to source code for more details.

## Credits

List your collaborators, if any, with links to their GitHub profiles.

If you used any third-party assets that require attribution, list the creators with links to their primary web presence in this section.

If you followed tutorials, include links to those here as well.

## License

The last section of a high-quality README file is the license. This lets other developers know what they can and cannot do with your project. If you need help choosing a license, refer to [https://choosealicense.com/](https://choosealicense.com/).
