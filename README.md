# Sudoku Solver App

This repo holds the code for an app to solve Sudoku for the user.
The purpose of this app is to have an interface where a user can upload an image of an unsolved Sudoku Puzzle and the deep learning model in this app will process the image to:  
1. Recognize the Sudoku Grid.  
2. Recognize the digits inside each box in the grid if any, else it will mark it as -1 (blank).  
3. Use recursion and backtracking algorithms to solve the puzzle.  

# Project Steps

I'm aware that there are various possible ways to make this project as I did some research after first trying to come up with the ideas by myself through brainstorming. In this section, I will try to walk you through the steps that I took to make this project. I first tried to wrap my head around the problems that I would need to solve.

We have two datasets:
1. Chars74k (only Digits)
2. TMNIST (Typeface MNIST)

I have modified both datasets as following:  
- The Chars74k dataset contains a total of 9,144 images 1,016 images for each digit from 1~9.
- The TMNIST dataset contains a total of 26,910 images 2,990 images for each digit from 1~9.

Modifications to the datasets:  
I have erased all the 0 digit images and pixel rows in both datasets for various reasons. We have to take into account that there is no need for our digit recognition model to recognize the digit '0' because we are creating a model specialized for Sudoku. As a reminder, Sudoku is a puzzle that involves only the digits from 1-9.  
I have only collected datasets such that the amount of images are equal for each digits because on my first attempt to doing this project, I noticed that during the softmax layer to classify a digit, when the model is unsure of the digit, it will be biased to predict the number with the most frequency in the training set. For my first I attempt the distribution of the dataset had been unbalanced with having more 1's than any other digit and so for digits that look similar such as 7 or 9 when the model was unsure it would predict 1.

p.s. Also, I am not including handwritten digits from datasets such as MNIST in this project because those digit formats are irrelevant to a Sudoku puzzle, which is mostly printed or made with computer fonts.

Some advantages of doing so:
1. by focusing on the digits that are relevant to the task, the model will be more specialized and better suited for the intended application.
2. it simplifies the training process and reduces the complexity of the model, leading to a faster training time and potentially better performance on the digits relevant to the task.
3. it helps balance the distribution of digits in the dataset preventing the model from being biased towards the majority class (1-9) and improve its ability to generalize.
4. reduce some bias and noise to the model.


# App Framework
streamlit

# Achnowledgements
Datasets:  
`Digit Recognition`  
> https://www.kaggle.com/datasets/karnikakapoor/digits  
Chars74k database by:  
This dataset contains digits in various digital fonts.

The Chars74K dataset is a specialized collection aimed at enhancing research in numeric character recognition. It encompasses a total of over 74,000 images, drawn from three distinct sources: 7,705 digits extracted from natural scenes, 3,410 digits hand-drawn using a tablet PC, and 62,992 digits synthesized from computer fonts. This dataset is organized into 10 classes, corresponding to the Hindu-Arabic numerals 0 through 9.

Crafted to support the development and testing of numeric recognition algorithms, the Chars74K dataset serves as a pivotal resource for tackling the complexities of digit recognition in varied contexts. Despite the high accuracy of character recognition systems in controlled settings, such as document scans with uniform fonts and backgrounds, the task remains challenging in less constrained environments captured by everyday photography and devices.

Introduced by T. E. de Campos, B. R. Babu, and M. Varma in their research on numeric character recognition in natural images at the International Conference on Computer Vision Theory and Applications (VISAPP) in Lisbon, Portugal, in February 2009, the Chars74K dataset is designed to address these real-world challenges. By providing a diverse array of digit images, it enables researchers and technologists to refine and advance the capabilities of automatic number recognition systems, ensuring they are robust and effective across a wide range of applications.  

>https://www.kaggle.com/datasets/nimishmagre/tmnist-typeface-mnist  
TMNIST (Typeface MNIST) by:  
TMNIST: A database of Typeface based digits

This dataset is inspired by the MNIST database for handwritten digits. It consists of images representing digits from 0-9 produced using 2,990 google fonts files.

The dataset consists of a single file:

TMNIST_Data.csv
This file consists of 29,900 examples with labels and font names. Each row contains 786 elements: the first element represents the font name (ex-Chivo-Italic, Sen-Bold), the second element represents the label (a number from 0-9) and the remaining 784 elements represent the grayscale pixel values (from 0-255) for the 28x28 pixel image.
NIMISH MAGRE

`Sudoku Box Detection`
> https://www.kaggle.com/datasets/macfooty/sudoku-box-detectionNewspaper clippings of sudoku boxes along with augmentation generated with Keras ImageGenerator.Augmentation on dataset from: https://icosys.ch/sudoku-dataset

`Sudoku Toy`  
> https://www.gmpuzzles.com/blog/2023/03/sudoku-by-thomas-snyder-20/  
The 435th puzzle from Thomas Snyder, aka Dr. Sudoku from the 2022 US Sudoku Grand Prix round.  