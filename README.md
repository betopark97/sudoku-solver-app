# Sudo Solver App

# Achnowledgements
Datasets:
> Digit Recognition
https://www.kaggle.com/datasets/pintowar/numerical-images
MNIST database by: 
Yann LeCun, Courant Institute, NYU
Corinna Cortes, Google Labs, New York
Christopher J.C. Burges, Microsoft Research, Redmond
http://yann.lecun.com/exdb/mnist/
+
Chars74k
J. J. Hull, A. Bharath, H. Bunke, "Document Image Database for the Evaluation of Recognition Methods", Proceedings of the International Conference on Document Analysis and Recognition (ICDAR), 1994. Available at: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
> Sudoku Box Detection
https://www.kaggle.com/datasets/macfooty/sudoku-box-detection
Newspaper clippings of sudoku boxes along with augmentation generated with Keras ImageGenerator.
Augmentation on dataset from: https://icosys.ch/sudoku-dataset
> Sudoku Toy
https://www.gmpuzzles.com/blog/2023/03/sudoku-by-thomas-snyder-20/
The 435th puzzle from Thomas Snyder, aka Dr. Sudoku from the 2022 US Sudoku Grand Prix round.


# Frameworks
streamlit

# Kaggle
https://www.kaggle.com/code/karnikakapoor/sudoku-solutions-from-image-computer-vision/notebook

https://www.kaggle.com/code/yashchoudhary/deep-sudoku-solver-multiple-approaches

https://www.kaggle.com/code/subhraneelpaul/sudoku-solver-using-tensorflow

https://www.kaggle.com/code/cdeotte/25-million-images-0-99757-mnist/notebook

# Youtube 
https://www.youtube.com/watch?v=G_UYXzGuqvM

https://www.youtube.com/watch?v=QR66rMS_ZfA


ToDo's:
1. Set up DVC to track data, MLFlow to track model
2. CNN to recognize sudoku boxes
3. CNN to recognize digits
4. Combine to output a 9x9 matrix
    - according digits and,
    - -1 for blank spaces
5. Algorithms to solve sudoku
    - backtracking
    - recursion
    - or use a neural network
    - compare to see which is faster
6. Make app
    - use streamlit




To make contour detection
Yes, you can indeed use contour detection as a technique to identify the outer border of the Sudoku grid and subsequently extract the individual cells containing the numbers. Here's a step-by-step approach you could follow:

Contour Detection: Apply contour detection algorithms (e.g., using OpenCV's findContours function) to identify the contours present in the input image. You can then filter the contours to find the largest contour, which represents the outer border of the Sudoku grid.

Approximate Polygon: Once you've identified the largest contour, you can approximate it as a polygon to simplify its shape. This step helps to reduce the complexity of the contour and make subsequent processing easier.

Perspective Transformation: Perform perspective transformation to rectify the Sudoku grid and align it to a regular grid layout. This step ensures that the cells in the grid are evenly spaced and aligned horizontally and vertically.

Cell Extraction: Divide the rectified Sudoku grid into individual cells by iterating over the grid and extracting each cell's region of interest (ROI). You can then further process each cell to extract the number or digit it contains using techniques such as image segmentation, thresholding, or OCR (Optical Character Recognition).

Number Recognition: Apply number recognition techniques to identify the digits present in each cell. This could involve using machine learning models, such as convolutional neural networks (CNNs), to classify the digits based on their visual appearance.

By following these steps, you can effectively use contour detection to identify the Sudoku grid's outer border and extract the individual cells containing the numbers. This approach can be simpler and more lightweight compared to using complex object detection models like Faster R-CNN, especially if your application's requirements are relatively straightforward. However, it's important to ensure robustness and accuracy by carefully tuning the contour detection parameters and handling various edge cases.