from comparer import compare
import filters
import numpy as np
from scipy.ndimage import rotate
from math import ceil, sqrt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def back_project_step(theta, sinogram, img_size, padding):
    # Take current sinogram column at angle theta
    # Reshape it to be a 2D array
    # Smear the projection by repeating the column across the entire width (at the same time adding horizontal padding)
    smear_img = np.repeat(np.reshape(sinogram[:, theta], (img_size, -1)), img_size+2*padding, axis=1)

    # Add horizontal padding for rotation
    smear_img = np.pad(smear_img, ((padding, padding), (0, 0)))

    # (theta+90)%360 -> Rotate the image by 90 degrees for alignment
    rotated = rotate(smear_img, (theta+90)%360, reshape=False)[padding:padding+img_size, padding:padding+img_size]

    return rotated
   

# The implementation follows Fabians implementation style
def back_project(sinogram):
    # Projection count is the width of the sinogram
    projection_count=sinogram.shape[1]
    # The image has is size of the height of the sinogram squared
    img_size=sinogram.shape[0]
    # Initialize the reconstructed image to all zero
    img = np.zeros((img_size, img_size))
    # Calculate die diagonal and according padding
    diagonal = ceil(sqrt(img_size**2 * 2))
    padding = ceil((diagonal - img_size) / 2)

    # Take sinogram colum at angle theta
    # put it in an image
    # extend the sinogram to the entire length
    # rotate that image
    # add that rotation to the already restored image
    for theta in range(projection_count):
       rotated = back_project_step(theta, sinogram, img_size, padding)
       img = np.add(img, rotated)

    return img
  

# filter being a function of filters
def apply_filter(sinogram, filter=filters.ramp):
  # Get the size of the image and number of angles
  img_dim, num_angles = sinogram.shape
  # Initialize the filtered sinogram to all zero
  filtered_sinogram = np.zeros(sinogram.shape)
  # Get the appropriate filter
  filter = filter(img_dim)

  # For every column
  # Fourier transform an shift it
  # Multiply by filter
  # Reverse fft
  # Insert filtered column into filtered sinogram
  for theta in range(num_angles):
    # Get the column
    col = sinogram[:,theta]

    # Fourier transform the column and shift the lower frequencies to the center
    col = np.fft.fft(col, axis=0)
    col = np.fft.fftshift(col)

    # Multiply the filter
    filtered_col = np.multiply(col, filter)

    # Reverse the shift and Fourier transform
    filtered_col = np.fft.ifftshift(filtered_col)
    filtered_col = np.fft.ifft(filtered_col, axis=0)

    # Insert it back into the now filtered sinogram
    filtered_sinogram[:,theta] = filtered_col

  # Since we get a complex array we need to transform it to a real array
  return np.real(filtered_sinogram)


def filtered_back_projection(sinogram, filter=filters.ramp):
    # Filter the sinogram
    filtered_sinogram = apply_filter(sinogram, filter)
    # Back-project the filtered sinogram
    return back_project(filtered_sinogram)

####### Plots #########################################################
def plot_applied_filter(sinogram, filter=filters.ram_lak):
   # Create sub plots
   fig, (ax_original, ax_filtered) = plt.subplots(1, 2, figsize=(13, 6))
   
   # Filter the sinogram
   filtered_sinogram = apply_filter(sinogram, filter)

   # Load the original unfiltered sinogram
   ax_original.imshow(sinogram, cmap="gray")
   ax_original.set_title("Original sinogram")

   # Load the filtered sinogram
   ax_filtered.imshow(filtered_sinogram, cmap="gray")
   ax_filtered.set_title(f"Filtered sinogram ({filter.__name__})")

   plt.show()

def compare_fbp_by_filter(image, sinogram):
   # Create subplot
   fig, axes = plt.subplots(2, 4, figsize=(13, 6))
   # Add spacing for the titles not to overlap
   fig.tight_layout(pad=5.0)
   # Define all filters
   filter_list = [filters.ramp, filters.ram_lak, filters.cosine, filters.hann]

   # List of all reconstructions, their filter and score
   reconstructions = []

   # We will store the reconstructions in a file, since it will take a long time to calculate them each time
   path = "recons.npy"
   try:
      reconstructions = np.load(path, allow_pickle=True)
   except OSError:
      for filter in filter_list:
         # Always store the reconstructions as a tuple
         # The reconstruction, used filter and the similarity score
         fbp_reconstruction = filtered_back_projection(sinogram, filter)
         reconstructions.append((fbp_reconstruction, filter, compare(image, fbp_reconstruction)))

      # Save the list to a file
      np.save(path, np.array(reconstructions, dtype=object))

   # For each filter and reconstruction
   for i in range(len(reconstructions)):
      # Get the reconstructed image, used filter and similarity score
      (reconstruction, filter_used, compare_value) = reconstructions[i]
   
      # Plot the filter
      axes[0, i].plot(filter_used(image.shape[0]))
      # Set the y-limits to 0 and 1 to have comparable graphs
      axes[0, i].set_ylim(0, 1)
      # Plot the reconstructed image
      axes[1, i].imshow(reconstruction, cmap="gray")
      # Set the title as the used filter and the similarity score
      axes[1, i].set_title(f"{str(filter_used.__name__)}\nScore: {compare_value:.2f}")

   plt.show()


####### Animation #####################################################
animate = False # Global variable to control whether or not to animate
if animate:
   # Load the sinogram from a file if it is available
   path = "sinogram.npy"
   try:
      sinogram = np.load(path)
      # If we want to plot the same for a filter -> filter the sinogram here
      # sinogram = apply_filter(sinogram, filters.ram_lak)
   except OSError:
      print("No sinogram found")
      exit()

   # Create the subplots
   fig, (ax_sinogram, ax_recon) = plt.subplots(1, 2, figsize=(13, 6))
   # Get information for the fbp (see above for details)
   img_size, num_angles = sinogram.shape
   diagonal = ceil(sqrt(img_size**2 * 2))
   padding = ceil((diagonal - img_size) / 2)

   # Initialize the reconstruction to zero
   reconstruction = np.zeros((img_size, img_size))

   def update(theta):
      # Modulo since we only have limited angles
      theta = theta % num_angles
      # We want to reference the global reconstruction not a local variable
      global reconstruction

      # Reset the image, if a full rotation is done
      if theta == 0:
         reconstruction.fill(0)

      # Clear the previous reconstruction process
      ax_recon.clear()

      # Get the current reconstruction step
      rotated = back_project_step(theta, sinogram, img_size, padding)
      # Add that step for current reconstruction (see above)
      reconstruction = np.add(reconstruction, rotated)

      # Plot the current reconstruction progress
      ax_recon.imshow(reconstruction, cmap='gray')
      ax_recon.set_title("Reconstruction")


   def init_animation():
      # We firstly load the sinogram since it will not be updated
      ax_sinogram.imshow(sinogram, cmap='gray')
      ax_sinogram.set_title("Sinogram")

   # Call the animation and show it
   animation = FuncAnimation(fig, update, frames=range(num_angles), interval=50, repeat=True, init_func=init_animation)
   plt.show()