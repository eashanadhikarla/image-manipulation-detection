
## Image Manipulation Detection

<p align="left", style="font-size:30px"><b>Author:</b><br />Eashan Adhikarla</p>

### Prerequisites
#### base
- matplotlib>=3.2.2
- numpy>=1.18.5
- opencv-python>=4.1.2
- Pillow
- PyYAML>=5.3.1
- scipy>=1.4.1
- torch>=1.7.0
- torchvision>=0.8.1
- tqdm>=4.41.0
#### logging 
- tensorboard>=2.4.1
#### plotting
- seaborn>=0.11.0
- pandas
#### export
- coremltools>=4.1
- onnx>=1.8.1
- scikit-learn==0.19.2  # for coreml quantization
#### extras
- thop  # FLOPS computation
- pycocotools>=2.0 

-----------------------------------------------------------------

### Dropbox Drive Link for Data & Output
(restricted to Lehigh email address)
All the trained models for Task 1 and Task 2 can be found in the 
link below.
- Link: (Task 1, Data) https://www.dropbox.com/sh/sl5p7ho1btrz0uk/AADIeMT3sd-OiyCSMaSygMsja?dl=0
- Link: (Task 2, Checkpoint) https://www.dropbox.com/sh/oavmxkb2ic1rx4e/AAClmYRUCXA1ZMCicGwPjLewa?dl=0
- Link: (Task 2, Data) https://www.dropbox.com/sh/oj6uiuoojvkqcrg/AAALudpDKtcu9PnJqnExGWvta?dl=0

-----------------------------------------------------------------

### File description:
#### Task 1:

![Detection on COCO Tampered Dataset](https://github.com/eashanadhikarla/image-manipulation-detection/blob/main/Task%201/output/test_batch1_pred.jpg)

- To detect the image manipulation localization detection using a
deep learning model of our choice.
- I chose to designed a state-of-the-art YOLO-V5 model for this task,
although there are other good choices too, such as, Mask-RCNN, 
Faster-RCNN, Efficient-D7, etc. Based on my previous experience with
YOLO-V5 it was the model of my choice.

Setup:
1. You can read the data using `dataloader.py`, it will read the path 
from the text files in coco_synthetic and copy the images into a new 
repository `./dataset/train/ & ./dataset/test/`. This is the first step 
for the data. 
2. To make the YOLO format, we need to normalize the pixel coordinate 
values to normalized values between 0 and 1. For this conversion just 
use the `annotations.py` script to perform the conversion and also 
generate a seperate annotation file for each image file. This is going 
to create a image-label pair (the way YOLO wants.)

(You can completely ignore that Step 1. and Step 2. and directly 
download the processed data from the link above in the dataset section.)

3. To run the program. You should have the dataset in the following structure:
    - dataset
        - images
            - train
            - val
        - labels
            - train
            - val
    - src/yolov5
        - data
            - forgery.yaml (edit the path here if needed)
        - train.py
        - test.py
        - (keep the rest of the files as it is)

4. Run the program:
    - First go to src/yolov5 directory and then run the following command

python3.8 train.py --img 640 --batch 16 --epochs 50 --data forgery.yaml --weights (downloaded from the link above)

### Results
#### YOLOv5x-TTA
![Precision-Recall Curve](https://github.com/eashanadhikarla/image-manipulation-detection/blob/main/Task%201/output/PR_curve.png)

#### Overall score
![Overall Metrics](https://github.com/eashanadhikarla/image-manipulation-detection/blob/main/Task%201/output/results.png)

#### Dataset Description
![Dataset COCO Tampered](https://github.com/eashanadhikarla/image-manipulation-detection/blob/main/Task%201/output/labels.jpg)

#### Task 2 & Bonus:
- To perform a deepfake detection on the face image dataset. As there are 
many face editing algorithms seem to produce realistic human faces, upon 
closer examination, they do exhibit artifacts in certain domains which 
are often hidden to the naked eye. 
- I implemented a novel architecture pipeline desinged by [Durall et al.] that 
uses four simple preprocessing/feature extraction steps and lastly performing 
classification on the extracted features.
- The steps are as follows:
    - Performing the Discrete Fourier Transform (DCT) on the image 
    - Convert it to Amplitude Spectrum 2D 
    - Perfrom the Azimulthal average over the amplitude 2D spectrum
    - Flatten the averaged data
    - Perform the classification of the choice.

Setup:
- The steps for this task is relatively simple and is as given below.

1. Copy the data and checkpoints from the link above to the `Task 2` folder or a working directory.
2. The directory should look like:
    - run.sh
    - FacesHQ.py
    - demo-task2.py
    - data/
        - FacesHQ_Data.pkl
        - Data.pkl
        - Bonus/
            - FaceHQ_traindata.pkl
            - FaceHQ_testdata.pkl
    - checkpoint/
        - svmBest.pt
        - mlpBest.pt
        - knnclassifierBest.pt
        - Bonus/
            - svmBest.pt
            - mlpBest.pt
            - knnclassifierBest.pt


-----------------------------------------------------------------

### Reference

- Dr. Aparna Bharati, Class notes, Presentation 6,7,10, (L6-Formats & Compression; L7-Image Tampering Detection; L10-Image Tampering Detection)
- Zhou et al., Learning Rich Features for Image Manipulation Detection, 
https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Learning_Rich_Features_CVPR_2018_paper.pdf
- Durall et al., Unmasking DeepFakes with simple Features, 
https://arxiv.org/pdf/1911.00686.pdf
- COCO Dataset, http://cocodataset.org/#download
- COCO PythonAPI, https://github.com/cocodataset/cocoapi

-----------------------------------------------------------------
