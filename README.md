# iCAD-Dental-Test
This repo is created as my solution to the i-CAD interview question

# Pre-processing
The dataset contains images of upper teeth and lower teeth. After looking at the images, the main distinction between the two categories was the starking difference in shape. Hence it made sense to extract the shape of these images. This was done using the following steps:

- **Histogram Equalisation**: In particular performed Adaptive histogram equalisation. This improved the contrast of the image
- **Thresholding**: Performed OTSU thresholding to get the boundry of the images

# Training
Once we get the external shape of the teeth, the next stage is to train the classifier. For this case I have chosen a pre-trained architecture and done transfer learning. This is because the dataset available is very small and hence it makes sense to perform trasnfer learning. The following steps were performed during training 

- **Basic Pre-processing and augmentation**: This step involved normalisation and resizing of the image. I have also performed data augmentation by rotating the image on horizontal axis. 
- **Early Stopping**: Since the dataset is small it makes sense to add early stopping. This is done to avoid overfitting of the data
- **Resume Training**: This step is not needed. However, it is a good practice to save checkpoints after a few epochs. This ensures that if our training suddenly stops due to unforseable reasons, our training would not start from scratch. As mentioned earlier this step is not needed for this dataset, since the dataset is small. However, I have added this as a good practice. 
- **Save best checkpoint**: I am saving the best checkpoint so that during testing we can just load the best checkpoint and run it on our test set. 

## Few Observations:
- I have not considered a validation dataset. This is because the dataset is very small. Creating a validation set would have reduced the size of training data which is not ideal. 
- I had trained my CNN model on raw data. However, it gave a very poor accuracy. Hence I decided to perform thresholding to extract just the shape of the teeth. I believe that the heatmaps might have tripped the CNN off. Once we get a larger dataset, the CNN might work well on raw dataset but for small dataset I forced the CNN to just focus on shape of the teeth. This gave me a good accuracy.   


# Results:
I have tested this dataset on a pretrained model(transfer learning) as well as custom architecture. The following were my results:


# Steps to recreate the code:






