# iCAD-Dental-Test
This repo is created as my solution to the i-CAD interview question

# Pre-processing
The dataset contains images of upper teeth and lower teeth. I have performed the following pre-processing techniques before traiing the model

 - **Normalisation**: The images are normalised. This helps the CNN while training 
 - **Histogram Equalisation**: The contrast of the image is improved. This is done because of the heatmaps present in the image. Improving the contrast helps CNN in identifying color differences between images. 
 - **Resizing of the image**: The image is resized to 416x416 and then passed on as an input to classifier.  

# Training
Once we get the external shape of the teeth, the next stage is to train the classifier. For this case I have chosen a pre-trained architecture and done transfer learning. This is because the dataset available is very small and hence it makes sense to perform trasnfer learning. The following steps were performed during training 

- **Custom Dataloader**: I have written my own custom dataloader to load he data. This is because the input data wasn't in the traditional pytorch format. 
- **Early Stopping**: Since the dataset is small it makes sense to add early stopping. This is done to avoid overfitting of the data
- **Save checkpoint**: I am saving the checkpoint after every epoch.  



# Results:
I have tested this dataset on a pretrained model(transfer learning) as well as custom architecture. The custom architecture did not give very good results. I believe thats because the transfer learning model already was pretrained on a significantly larger dataset. Hence transfer learning out performed custom architecture.


# Steps to recreate the code:
- To recreate the code, just mention the path to your image directory on line 29. The path should be "some_path/Imagefiles/". Make sure that you add the leading "/" at the end.







