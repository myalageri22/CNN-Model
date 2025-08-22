AI Model for Binary Image Segmentation of Vessel Networks in the Heart based on patient MRI scans

Model Information:

Datasets/Preprocessing:

The neural network will be trained on MRI segmented images from the HVSMR 2.0 Dataset, and these images will be converted from DICOM to NifTI using the Python library, 
Nibabel, and will be preprocessed for normalization, resizing, denoising, and augmentation. 

Link to Dataset: https://figshare.com/articles/dataset/HVSMR-2_0_orig_/25226360?backTo=/collections/HVSMR-2_0_A_3D_cardiovascular_MR_dataset_for_whole-heart_segmentation_in_congenital_heart_disease/7074755

Model Structure:

The algorithm includes a 3D U-Net CNN (Convolutional Neural Network), which is specialized for image segmentation pipelines and will contain an 
encoder, bottleneck, decoder, and finally the output layer. The input of the engine will be the MRI scans, and the output will be a segmentation mask, 
highlighting the vascular structures into a highly detailed map of vessel and non-vessel using the frameworks, PyTorch and MONAI. 

Model Training:

Supervised learning will be the method employed with 70% of the dataset for training, 15% for testing, and 15% validation. 
Due to the large size of 3D MRI scans, batch sizes will range from 1–4. 
The network will be trained for approximately 100 epochs using the Adam optimizer. For model evaluation, the Dice coefficient 
will be used to measure the accuracy between the prediction of the AI and the ground truth, and Hausdorff distance to measure 
the distance between the predicted and the actual vessel boundaries. 

(Model Parameters will be adjusted during actual training)

Post-processing and STL File Conversion:

After segmentation is complete, steps will be taken to clean up the mask, the output, and then converted into 3D meshes 
and then exported as a STL file with smoothing algorithms and even a constraint reinforcement to set the parameters using 
tools such as 3D slicer and Blender to convert it into a 3D Bio Printable format. 

- Note this part still needs to be implemented. 
