# Building Detection in Satellite Images

## Project Overview

### Introduction
This project focuses on analyzing satellite images to detect and mark buildings within these images. Using advanced data science techniques, the project automatically generates masks over the satellite images, highlighting the locations of the buildings.

### Objective
The primary objective of this project is to aid in disaster risk management by digitizing buildings in satellite imagery. This process is a crucial first step in humanitarian aid, as it helps identify which areas require more assistance in the aftermath of a disaster. While it might seem that most places are already mapped, the reality is different. Many regions, especially in developing countries, remain unmapped. These areas are often more vulnerable to disasters due to less resilient infrastructure. By providing accurate and up-to-date maps of these regions, our project aims to enhance disaster preparedness and response, ultimately contributing to saving lives and reducing the impact of disasters.  


## Methodology

### Data
The data for this project was sourced from the AIcrowd Mapping Challenge, available at [AIcrowd Mapping Challenge](https://www.aicrowd.com/challenges/mapping-challenge). This challenge provides a comprehensive dataset specifically designed for training and validating models aimed at satellite image analysis and building detection.  
  
#### Dataset Composition
The dataset consists of two main parts:

1. **Training Set (`train.tar.gz`):** This set includes 280,741 tiles, each being a 300x300 pixel RGB image of satellite imagery. These images are accompanied by annotations in the MS-COCO format, providing detailed information about the building locations within each tile.

2. **Validation Set (`val.tar.gz`):** The validation set comprises 60,317 tiles, also in the format of 300x300 pixel RGB satellite images. Similar to the training set, these images include corresponding annotations in MS-COCO format, allowing for effective model validation.

This dataset is instrumental in training our model to accurately detect and mark buildings in satellite imagery. The extensive number of images and the precision of the annotations provide a robust foundation for developing a model that can reliably identify buildings, which is essential for applications in disaster risk management and humanitarian aid.  

### Model Architecture
#### Baseline Model (Unet)
I use a Unet for the baseline model. The U-Net model architecture consists of an encoder-decoder structure, designed to capture intricate features and spatial information for effective image segmentation. The input layer receives a 256x256x3 image, and the encoder, composed of convolutional layers, progressively extracts hierarchical features. Each encoder layer is followed by batch normalization and activation functions, enhancing learning stability. Max-pooling layers reduce spatial dimensions, while the decoder, with transposed convolutional layers, upscales the spatial information. Crucially, skip connections concatenate encoder output to corresponding decoder input, preserving fine-grained details. The final layer outputs a 256x256x1 mask. With approximately 7.7 million parameters, the model balances complexity and efficiency. This architecture enables the U-Net to excel in tasks like building detection in satellite imagery, combining semantic understanding with precise spatial localization.

##### Training
I trained the baseline model for 5 epochs only since the dataset is large and the training process is highly time consuming.  

##### Results  
![acc1.png](https://github.com/alihijazy/building-recognition-in-satellite-images/blob/master/images/acc1.png)
![prec1.png](https://github.com/alihijazy/building-recognition-in-satellite-images/blob/master/images/prec1.png)
![loss1.png](https://github.com/alihijazy/building-recognition-in-satellite-images/blob/master/images/loss1.png)  

##### Performance
below are the scores if the baseline model:  
- Training precision: 0.94
- Validation precision: 0.93
- Training accuracy: 0.97
- Validation accuracy: 0.97
- Training loss: 0.075
- Validation loss: 0.081  
  
##### Examples
![pred_baseline.png](https://github.com/alihijazy/building-recognition-in-satellite-images/blob/master/images/pred_baseline.png)
![pred_baseline_2.png](https://github.com/alihijazy/building-recognition-in-satellite-images/blob/master/images/pred_baseline_2.png))


#### 2nd Model (Pretrained ResNet-34)
The second model is a ResNet-34 pretrained on general images and fine-tuned for your segmentation task. With a 256x256x3 input, it utilizes the ResNet's deep architecture and skip connections for robust feature extraction. The retraining process tailors the model to your data, balancing the pretrained knowledge with dataset-specific details. The final layer produces a 256x256x1 mask, making it adept at tasks like building detection in satellite imagery, combining semantic understanding with precise spatial localization.

##### Training
I trained the baseline model for 10 epochs only since the dataset is large and the training process is highly time consuming.

##### Results
![acc-prec2.png](https://github.com/alihijazy/building-recognition-in-satellite-images/blob/master/images/acc-prec2.png)
![loss2.png](https://github.com/alihijazy/building-recognition-in-satellite-images/blob/master/images/loss2.png)
  
##### Performance
below are the scores if the 2nd model:  
- Training precision: 0.96
- Validation precision: 0.96
- Training accuracy: 0.98
- Validation accuracy: 0.98
- Training loss: 0.045
- Validation loss: 0.047

##### Examples
![pred_pretrained.png](https://github.com/alihijazy/building-recognition-in-satellite-images/blob/master/images/pred_pretrained.png)
![pred_pretrained_2.png](https://github.com/alihijazy/building-recognition-in-satellite-images/blob/master/images/pred_pretrained_2.png)

## Deployment  
As a deployment tool I created a local streamlit page where you can paste any coordinates and press download and process, and you will get the mask and the image of the entered coordinates. below is a sample:  
![image and mask.png](https://github.com/alihijazy/building-recognition-in-satellite-images/blob/master/images/image%20and%20mask.png)
Then this masked is plugged to any GIS software and transformed from raster(image) to a vector file (shapefile) which is a very easy to handle format to fix the errors within the mask. below are the results:  
![adjusted masks.png](https://github.com/alihijazy/building-recognition-in-satellite-images/blob/master/images/adjusted%20masks.png)

## Future Enhancements
In the upcoming phases of our project, we aim to enhance the efficiency of the georeferencing process for downloaded Bing satellite images. Our goal is to introduce an automated georeferencing module, streamlining the task for users.

This enhancement will eliminate the need for manual georeferencing efforts. The proposed module will either tap into the geographic information embedded in the image metadata or employ advanced image recognition algorithms. These algorithms will identify distinct features, such as road intersections or prominent landmarks, aiding in the precise spatial alignment of the image.

By automating the georeferencing step, we anticipate significant time savings for users. This enhancement ensures that subsequent GIS operations can rely on accurate geographical referencing. As a result, our users can focus more on the digitization process, knowing that the georeferencing aspect is seamlessly handled by the system.



