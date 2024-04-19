import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Preprocessing the Sudoku Grid
def preprocess_grid(image:np.array, resize_dim:tuple=(450,450)) -> np.array:
    """
    A function that preprocesses an image by converting it to grayscale, resizing it, and applying Gaussian blur.
    
    Parameters:
        image (np.array): The input image to be processed.
        
    Returns:
        np.array: The preprocessed image as a NumPy array.
    """
    height = resize_dim[0]
    width = resize_dim[1]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (height,width))
    blurred_image = cv2.GaussianBlur(resized_image, (5,5), cv2.BORDER_DEFAULT)
    final_image = blurred_image
    return final_image

def get_contour_corners(image:np.array) -> np.array:
    """
    A function that takes an image as input and returns the corners of the largest contour in the image.
    
    Parameters:
        image (np.array): The input image as a NumPy array.
        
    Returns:
        np.array: The corners of the largest contour in the image as a NumPy array.
    """
    threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.1 * cv2.arcLength(largest_contour, True)
    contour_corners = cv2.approxPolyDP(largest_contour, epsilon, True)
    return contour_corners

def sort_corners(corners:np.array) -> np.array:
    """
    Sorts the corners in clockwise order starting from the top-left corner.
    """
    sorted_corners = np.zeros((4,2), dtype=np.float32)
    corners = corners.reshape((4,2))
    sorted_corners[0] = corners[np.argmin(np.sum(corners, axis=1))]
    sorted_corners[1] = corners[np.argmin(np.diff(corners, axis=1))] 
    sorted_corners[2] = corners[np.argmax(np.sum(corners, axis=1))]
    sorted_corners[3] = corners[np.argmax(np.diff(corners, axis=1))]
    return np.float32(sorted_corners)

def target_corners(image:np.array) -> np.array:
    """
    A function that calculates the corners of the target rectangle based on the input image dimensions.
    
    Parameters:
        image (np.array): The input image as a NumPy array.
        
    Returns:
        np.array: The corners of the target rectangle as a NumPy array.
    """
    resize_dim = (image.shape[0], image.shape[1])
    height, width = resize_dim
    target_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    return np.float32(target_corners)

def warp_image(image:np.array, sorted_corners:np.array, targeted_corners:np.array) -> np.array:
    """
    Warps an image using perspective transformation based on the given sorted corners and targeted corners.
    Args:
        image (np.array): The input image as a NumPy array.
        sorted_corners (np.array): The sorted corners of the image as a NumPy array.
        targeted_corners (np.array): The target corners for the perspective transformation as a NumPy array.
    Returns:
        np.array: The warped image as a NumPy array.
    """
    perspective_matrix = cv2.getPerspectiveTransform(sorted_corners, targeted_corners)
    warped_image = cv2.warpPerspective(image, perspective_matrix, image.shape[:2])
    return warped_image

def divide_into_9x9_boxes(sudoku_grid:np.array) -> list:
    """
    Divides a Sudoku grid into 9x9 boxes and returns a list of these cropped boxes.
    Args:
        sudoku_grid (np.array): The input Sudoku grid as a NumPy array.
    Returns:
        list: A list containing the 9x9 cropped boxes from the Sudoku grid.
    """
    # Get image dimensions
    height, width = sudoku_grid.shape[:2]

    # Calculate box size
    box_height = height // 9
    box_width = width // 9

    # Initialize list to store cropped boxes
    boxes = []

    # Iterate over the rows and columns to extract 9x9 boxes
    for i in range(9):
        for j in range(9):
            # Calculate coordinates for each box
            y_start = i * box_height
            y_end = (i + 1) * box_height
            x_start = j * box_width
            x_end = (j + 1) * box_width

            # Crop box from image
            box = sudoku_grid[y_start:y_end, x_start:x_end]
            boxes.append(box)

    return boxes

def crop_boxes(box_images:list) -> list:
    """
    Given a list of box images, this function crops each image to a specified region and returns a list of the cropped images.
    
    :param box_images: A list of images representing sudoku boxes.
    :type box_images: list
    :return: A list of cropped images.
    :rtype: list
    """
    left = 3
    upper = 3
    right = 47
    lower = 47
    cropped_boxes = []
    for box in box_images:
        # Crop the image
        box = Image.fromarray(box)
        cropped_image = box.crop((left, upper, right, lower))
        cropped_boxes.append(cropped_image)
    return cropped_boxes

def plot_sudoku_boxes(box_images:list) -> None:
    """
    Plots sudoku boxes using the input list of box images.
    
    Args:
        box_images (list): A list of images representing sudoku boxes.
    
    Returns:
        None
    """
    fig, axs = plt.subplots(9, 9, figsize=(3, 3))

    # Iterate over each subplot and image
    for ax, image in zip(axs.ravel(), box_images):
        ax.imshow(image, cmap='gray')  # Display the image in grayscale
        ax.axis('off')  # Turn off axis labels and ticks

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()



# Preprocess for CNN
def image_to_pixels(image:np.array, resize_dim:tuple=(32,32)) -> np.array:
    """
    A function that converts an image to grayscale, equalizes it, resizes it, and returns the processed image as a NumPy array.
    Parameters:
        image (np.array): The input image to be processed.
    Returns:
        np.array: The processed image as a NumPy array.
    """
    if len(image.shape) >= 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)
        image = cv2.resize(equalized_image, resize_dim)
    else:
        print('Already grayscale')
        equalized_image = cv2.equalizeHist(image)
        image = cv2.resize(equalized_image, resize_dim)
    return image

def preprocess_pixels(image:np.array, resize_dim:tuple=(32,32)) -> np.array:
    """
    Resizes and normalizes an image and returns the processed image as a NumPy array.
    Parameters:
        image (np.array): The input image to be preprocessed.
        
    Returns:
        np.array: The preprocessed image as a NumPy array.
    """
    normalized_image = tf.keras.utils.normalize(image, axis=1)
    preprocessed_image = np.array(normalized_image).reshape(-1, resize_dim[0], resize_dim[1], 1)
    return preprocessed_image


def predict_sudoku_grid(box_images:list, model:tf.keras.Model, verbose:int=0) -> np.array:
    """
    Predicts the values of a Sudoku grid based on a list of box images using a trained model.
    Args:
        box_images (list): A list of images representing Sudoku boxes.
        model (tf.keras.Model): The trained model used for prediction.
        verbose (int, optional): If set to 1, prints the prediction for each box. Defaults to 0.
    Returns:
        np.array: A 9x9 NumPy array representing the predicted Sudoku grid.
    """
    sudoku_grid = np.zeros(81)

    for index, box in enumerate(box_images):
        box = np.array(box)
        box = cv2.resize(box, (28,28))
        box = preprocess_pixels(box)

        prediction = model.predict(box, verbose=0)
        if verbose == 1:
            print(f'Box {index+1}: {prediction.round(3)}')
        predicted_probability = prediction.max()
        if predicted_probability < .6:
            prediction_label = 0
        else:
            prediction_label = prediction.argmax() + 1
            
        sudoku_grid[index] = prediction_label
    return sudoku_grid.reshape((9,9))

# Solve Sudoku
def possible(y, x, n, sudoku_grid):
    for i in range(9):
        if sudoku_grid[i][x] == n:
            return False
    for i in range(9):
        if sudoku_grid[y][i] == n:
            return False
    x0 = (x//3) * 3
    y0 = (y//3) * 3
    for i in range(3):
        for j in range(3):
            if sudoku_grid[y0+i][x0+j] == n:
                return False
    return True

def solve_sudoku(sudoku_grid):
    for y in range(9):
        for x in range(9):
            if sudoku_grid[y][x] == 0:
                for n in range(1, 10):
                    if possible(y, x, n, sudoku_grid):
                        sudoku_grid[y][x] = n
                        solve_sudoku(sudoku_grid)
                        sudoku_grid[y][x] = 0
                return
    print(sudoku_grid)