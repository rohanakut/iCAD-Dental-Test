# iCAD-Dental-Test
This repo is created as my solution to the i-CAD interview question

# Pre-processing
The dataset contains images of upper teeth and lower teeth. After looking at the images, the main distinction between the two categories was the starking difference in shape. Hence it made sense to extract the shape of these images. This was done using the following steps:

- **Histogram Equalisation**: In particular performed Adaptive histogram equalisation. This improved the contrast of the image
- **Thresholding**: Performed OTSU thresholding to get the boundry of the images


