# VRDL_HW3_Instance_Segmentation
Nuclear segmentation dataset contains 24 training images with 14,598 nuclear and 6 test images with 2,360 nuclear.
Train an instance segmentation model to detect and segment all the nuclei in the image.
No external data should be used!

## Coding Environment
- Jupyter Notebook

## Reproducing Submission
To reproduct the testing prediction, please follow the steps below:
1. [Jupyter Notebook environment](#environment)
2. [Dataset](#dataset)
3. [Training](#training)
4. [Testing](#testing)

## Environment
requirement.txt contains all packages version of Jupyter Notebook
- notebook 6.1.5  

## Dataset
- “Transfer_mask_To_json.ipynb” which can transfer mask image file to .json file. 
- I transfer the file on google colab and save on the google drive , then download the json file to my computer to train the model.
1. Download the dataset from https://drive.google.com/file/d/1nEJ7NTtHcCHNQqUXaoPk55VH3Uwh4QGG/view.
2. Upload unzipped file "dataset.zip"、"Transfer_mat_To_csv.ipynb" to google drive in the same folder.
3. Run the file "Transfer_mat_To_csv.ipynb" you will get a "annotations.json" file.
4. Download the file "annotations.json".



## Training
- Download the dataset which is uploaded by me and put it in the same file with "VRDL_HW03.ipynb" and "annotations.json".
- Run the files "VRDL_HW03.ipynb" will start to train the model and save it as "model_final.pth".
- Remember to replace the root of the image file with your own root.

The training parameters are:

Model | learning rate | Training iterations | Batch size
------------------------ | ------------------------- | ------------------------- | -------------------------
MaskRCNN_resnet101_fpn | 0.00025 | 100000 | 2

## Testing
- "VRDL_HW03.ipynb" has the code that can use the model which is saved above to predict the testing images and save the prediction result as json files according to coco set rules.

### Pretrained models
Pretrained model "MaskRCNN_resnet101_fpn" which is provided by detectron2.

### Link of my trained model
- The model which training with 100000 iterations：https://drive.google.com/file/d/14FpmiZiJ1SdBGayvGkJAg-5zSRXWL0jq/view?usp=sharing
- The model's training json file :https://drive.google.com/file/d/1Gp5-SdGiGUjhDb22cIQgHC0VIneMSgXY/view?usp=sharing

### Inference

Load the trained model parameters without retraining again.

“model_final.pth” need to be download to your own device and run “inference.ipynb” you will get the results as json file.
“model_final.pth” need to be put in the folder ./output/ that contains “inference.ipynb”.
