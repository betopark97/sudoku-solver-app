# Sudoku Solver App

This repo holds the code for an app to solve Sudoku for the user.
The purpose of this app is to have an interface where a user can upload an image of an unsolved Sudoku Puzzle and the deep learning model in this app will process the image to: 
    (1) Recognize the Sudoku Grid.
    (2) recognize the digits inside each box in the grid if any, else it will mark it as -1.
    (3) Use recursion and backtracking algorithms to solve the puzzle.

# App Framework
streamlit

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