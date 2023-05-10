from numpy import subtract, multiply, sum
from math import sqrt, log

# This method takes an original image and a reconstruction and gives a similarity value
def compare(original, reconstruction):
    # Subtract the images to get the difference
    difference = subtract(original, reconstruction)
    # Make it positive by squaring it
    difference = multiply(difference, difference)

    # Sum the error
    error_sum = sum(difference.flatten())

    # Return the total error as the square root of the error
    return sqrt(error_sum)