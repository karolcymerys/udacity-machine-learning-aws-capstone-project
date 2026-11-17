# Inventory Monitoring at Distribution Centers

## Introduction

One of the biggest challenges in distribution centers is inventory level monitoring 
and verification whether shipments contain appropriate number of items. 
Imagine a situation when customers receive incomplete package or when the logistics department discovers 
that products are missing just before assembling a shipment for dispatch.
In both cases it's waste of time and money. On the other hand, manual verification of shipments 
or inventories level is not also cheap and easy task. 
Automation of the process would allow to save both time and money for distribution centers. 
Therefore, this project aims to solve this problem.

## Technical problem statement

From technical point of view, aforementioned problem concerns object counting. 
Solution to be built should provide number of items based on provided input.
That number could be entered to inventory system, so it can be easily tracked or compared with purchase statement.
The most natural and handy form of mentioned input is image. 
Therefore, **this project aims to build vision model to count items on provided image**.

### Dataset

In order to prepare vision model, there is a need to train and test such model on images of bins that contain items.
One of the datasets that could be employed is [Amazon Bin Image Dataset [1]](https://registry.opendata.aws/amazon-bin-imagery/).
It's publicly available dataset distributed under *Creative Commons Attribution-NonCommercial-ShareAlike 3.0 United States (CC BY-NC-SA 3.0 US)* licence.
Dataset itself contains over 500 000 images of bins with items captured during normal operations in *Amazon Fulfillment Center*.
Additionally, for each photo dataset provides JSON file with metadata containing number of items in bins and characteristics of these items (like: name, size, weights, etc.).

Dataset can be easily obtained from publicly available S3 Bucket **aft-vbi-pds**.

### Proposed solution

Since the aftermentioned dataset for each image provides only number of items in bins, then this problem can be recognized as a **regression problem**.
Proposed solution assumes using backbone of pre-trained vision model **ResNet-101** [2]
and replacing classification head with a single dense layer yielding a estimated number of items in bin.
Since backbone of **ResNet-101** is already able to extract features from provided image, therefore only the last layer is going to be trained (fine-tuning).
A MAE is going to be employed as optimization criterion. For inference returned value will be rounded to nearest integer.

To implement this solution *PyTorch* framework and *AWS SageMaker* platform will be employed:
- *Amazon Simple Storage Service (S3)* will be used to store dataset
- *SageMaker Studio* will be used as web-based IDE
- *SageMaker Training Jobs* for models' training
- *SageMaker Real-time endpoint* and *SageMaker Batch Transform* for inference

### Considered alternatives

As an alternative solution, **instance segmentation** models were considered. 
In this approach we could employ one of pre-trained instance segmentation models like **Mask R-CNN** [3], **YOLO26-Seg**  [4] or **SAM** [5] (Segment Anything Model).
These models as input expect image from dataset and as response it provides segmentation binary mask and confidence score for each detected object.
Based on that it's possible to provide number of items on provided items. 
Since, *Amazon Bin Image Dataset* does not contain segmentation masks, it's not possible to train these models while using this dataset.
Therefore, we would need to solely rely on pre-trained model. However, this approach may struggle to provide accurate estimates, 
because characteristics of *Amazon Bin Image Dataset* may significantly differ from datasets that were use to pre-train segmentation models.

### Evaluation

In order to evaluate performance of solution, there is a need to split dataset into (1) train, (2) validation and (3) test subset with proportion 70:15:15.
That will require randomly but uniformly splitting dataset across classes (number of items in bin). 
That will guarantee that we have sufficient data to train and test model.
Performance itself will be measured by Mean Absolute Error.

### Benchmark model

As a benchmark model, a lighter version of the ResNet architecture will be used - **ResNet-18**. 
It will be adjusted in a similar way to solve the regression problem of estimating the number of items in a provided image.
Using **ResNet-18** as a baseline model will make it possible to evaluate whether the increased depth of ResNet-101 yields to get measurable performance gains over a lighter architecture.

## Summary
In this document solution for inventory level monitoring problem was discussed. 
Problem is considered as regression problem. **ResNet-101** was selected as vision model that is supposed to 
process images from *Amazon Bin Image Dataset* and returns estimated number of items in bin on provided image.
Performance of built model is going to be evaluated on extracted test image set from *Amazon Bin Image Dataset* before training.
As evaluation metric, MAE is going to be employed.

---

[1] *Amazon Bin Image Dataset* was accessed on *2026-11-17* from https://registry.opendata.aws/amazon-bin-imagery  

[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: *Deep Residual Learning for Image Recognition*, https://doi.org/10.48550/arXiv.1512.03385  

[3] Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick: *Mask R-CNN*, https://doi.org/10.48550/arXiv.1703.06870  

[4] Ranjan Sapkota, Rahul Harsha Cheppally, Ajay Sharda, Manoj Karkee: *YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection*, https://doi.org/10.48550/arXiv.2509.25164  

[5] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick: *Segment Anything*, https://doi.org/10.48550/arXiv.2304.02643

