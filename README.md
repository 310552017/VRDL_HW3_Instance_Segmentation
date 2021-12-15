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
- The model which training with 100000 iterations：https://drive.google.com/drive/folders/1g4r5g5v9L76khoL96FINSCno_9ZsRwMj?usp=sharing

### Inference

Load the trained model parameters without retraining again.

“utils.py”、“transforms.py”、“coco_eval.py”、“model_utils.py”、“engine.py”、“coco_utils.py” and ".pth" need to be download to your own device and run "VRDL_HW2_train.ipynb" you will get the results as json file.

"Inference.ipynb" just has the code about calculating the running time of the model.

"Inference.ipynb" and the model need to be upload to google colab to run so that it can has the same hard device performance.
