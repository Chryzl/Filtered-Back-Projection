import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import rotate
import math

def plot_fourier_slice_theorem(image, sinogram):
    ft_img, ft_sin, rotated_data = fourier_slicer(image, sinogram)

    # Create a 2D grid of data for the images
    img_height, img_width = image.shape
    x_img = np.linspace(-5, 5, img_width)
    y_img = np.linspace(-5, 5, img_height)
    X_img, Y_img = np.meshgrid(x_img, y_img)

    # Create a 2D grid of data for the sinogram
    sin_height, sin_width = sinogram.shape
    x_sin = np.linspace(-5, 5, sin_width)
    y_sin = np.linspace(-5, 5, sin_height)
    X_sin, Y_sin = np.meshgrid(x_sin, y_sin)

    # create figure
    fig = plt.figure(figsize=(13,7))

    # add the 2D fourier transformed image to the plot
    ax_img = fig.add_subplot(131, projection='3d')
    ax_img.plot_surface(X_img, Y_img, np.abs(ft_img), cmap='viridis')
    ax_img.set_title("2D Fourier transformed image", wrap=True)

    # add the individually 1D transformed sinogram columns
    ax_sin = fig.add_subplot(132, projection='3d')
    ax_sin.plot_surface(X_sin, Y_sin, np.abs(ft_sin), cmap='viridis')
    ax_sin.set_title("1D Fourier transformed sinogram", wrap=True)

    # add the 4 1D transformed sinogram columns into 2D Fourier space
    ax_recon = fig.add_subplot(133, projection='3d')
    ax_recon.set_title("Aligned sinogram in 2D Fourier space", wrap=True)
    for aligned_column in rotated_data:
        ax_recon.plot_surface(X_img, Y_img, np.abs(aligned_column), cmap='viridis')

    plt.show()

def fourier_slicer(image, sinogram):
    img_height, img_width = image.shape

    # 2D Fourier Transformed image
    ft_img = np.fft.fft2(image)
    ft_img = np.fft.fftshift(ft_img)

    # 1D Fourier Transformed sinogram
    ft_sin = np.fft.fft(sinogram, axis=0)
    ft_sin = np.fft.fftshift(ft_sin)
    # Both the sinogram and image will be shifted (fftshift) to align the lower frequencies in the center

    # For each rotated colum we will generate a new image to be stored in this list
    rotated_data = []
    # Rotate every 90° sinogram colum
    for theta in range(0, 360, 90):
        # Get the according colum and reshape to 2D array
        col = np.reshape(ft_sin[:,theta], (-1, img_width))

        # Pad the 2D array for rotation
        rot_col = np.pad(col, ((img_width // 2, img_width - img_width // 2 - 1), (0,0)))

        # Rotate the 2D array to aligne it at angle theta
        rot_col = rotate(rot_col, theta, reshape=False)

        # Interestingly the factor is lost, when the data is not inserted
        #rot_col = np.divide(rot_col, math.sqrt(2*math.pi))

        # Add the resulting 2D array
        rotated_data.append(rot_col)

    # Return everything
    return ft_img, ft_sin, rotated_data
