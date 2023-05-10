# Filtered Back Projection

I have written this code as part of a university seminar at TU Munich. It implements a very basic filtered-back-projection and visualizes the way it works.
It also includes a visualization of the Fourier slice theorem.

---

# Run

### Animations / Plots
1. (Filtered)-Back-Projection:

    The fbp.py file has a section at the bottom labeled as Animation. The very first variable has to be set to true, in order for the animation to work.

    ``animate = True``

    This should only be true, when the animation is wanted. If it is set to true and the file is loaded somewhere else, it will start running.

    For animating a filtered version, go to the animate section in the fbp.py file. There un comment the line below. It can be found in the loading section.

    ``sinogram = apply_filter(sinogram, filters.ram_lak)``

    ### Run
    For the animation to start, just run the file

    ### Troubleshooting
    There has to be a sinogram.py file beforehand. If it is not yet there, just run the init.py file first, it will create a file if necessary

2. Fourier Slice Theorem
   
   Just uncomment the line

   ``show_fourier()``

   in the init.py file and run the init.py file

3. Show the difference between a filtered and an unfiltered sinogram
   
   Just uncomment the line

   ``fbp.plot_applied_filter(sinogram)``

   in the init.py file and run the init.py file. It will take a sinogram and any filter from the filters.py file.

4. Display all filters
   
   Just uncomment the line

   ``filters.plot_all()``

   in the init.py file and run the init.py file

5. Compare the results of the different filters

   Just uncomment the line

   ``fbp.compare_fbp_by_filter(image, sinogram)``

   in the init.py file and run the init.py file. It takes any image and its sinogram. The image is used to calculate a similarity score.
