import fbp
import fourier_slice
import comparer
import filters
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon

# Define file name for sinogram
path = "sinogram.npy"

# Load Shepp-Logan phantom
image = shepp_logan_phantom()

def main():
    try:
        sinogram = np.load(path)
    except OSError:
        sinogram = radon(image, range(360))
        np.save(path, sinogram)

    # show_fourier()

    # Run animation for BP in fbp.py

    # fbp.plot_applied_filter(sinogram)

    # filters.plot_all()

    # fbp.compare_fbp_by_filter(image, sinogram)

def show_fourier():
    # square in the middle
    image = np.zeros((400,400))
    image[175:225, 175:225] = 1

    sinogram = radon(image, range(360))

    fourier_slice.plot_fourier_slice_theorem(image, sinogram)


if __name__ == "__main__":
    main()