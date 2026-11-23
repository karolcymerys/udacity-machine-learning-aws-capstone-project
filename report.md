# Inventory Monitoring at Distribution Centers

## Problem definition

One of the biggest challenges in distribution centers is inventory level monitoring. 
Lack of knowledge about current goods level can lead to delivery delays and incur additional costs for distribution centers. 
This project aims to address this problem by preparing solution that would allow to automate this process.
It proposes machine learning based solution that predicts number of items in bin on provided image.

## Dataset

This project utilizes [Amazon Bin Image Dataset [1]](https://registry.opendata.aws/amazon-bin-imagery/).
It's publicly available dataset distributed under *Creative Commons Attribution-NonCommercial-ShareAlike 3.0 United States (CC BY-NC-SA 3.0 US)* licence.
Dataset contains over 500 000 images of bins with items captured during normal operations in *Amazon Fulfillment Center*.
Additionally, for each image manifest file is supplemented that provides metadata: number of items in bins and other characteristics, like: name, size, weights, etc.
Dataset can be easily obtained from publicly available S3 Bucket **aft-vbi-pds**.

Performed analysis reveled that dataset contains 536 4343 images and number of items in bins ranges from 0 objects to 209 objects.
Below plot depicts that number of items in bins in not uniformly distributed: dataset contains much more instances of images in which number of items in bins is quite low (below 50) than for bins with a lot of objects in bins.

![Class distribution](assets/class_distribution.png)

For training and evaluation purposes dataset was split into three dataset with below proportions:
* training subset **70 %**
* validation subset **15 %**
* test subset **15 %**

All these subset were created once and kept fixed for all proposed solutions.

Images were randomly assigned to subsets, but proportions across number of items are remained: 
for each number of items in bins training subset contain 70%, validation 15% and test 15% of images for that number of items in bins.

On provided images following transformation were performed:
* training:
  * resizing to 224x224 (since that's requirement of ResNet models)
  * randomly horizontally and vertically flipped for data augmentation
  * applied Colour Jittering for brightness, contrast and saturation insensitivity
  * normalized across RGB-channels with following mean-std pairs
    * R channel: mean: 0.485, std: 0.229
    * G channel: mean: 0.456, std: 0.224
    * B channel: mean: 0.406, std: 0.225
* testing:
  * resizing to 224x224 (since that's requirement of ResNet models)
  * normalized across RGB-channels with following mean-std pairs
    * R channel: mean: 0.485, std: 0.229
    * G channel: mean: 0.456, std: 0.224
    * B channel: mean: 0.406, std: 0.225

## Solutions

From technical point of view, problem to be solved is a regression problem - for provided image solution needs to provide number of items detected in bin.
Since this problem requires image processing, the best option is to employ one of popular pre-trained vision models.

As a foundation for both of proposed solutions ResNet architecture was selected [2]. 
It's one of the most popular vision model designed to solve image classification problems. 
Its characteristics - residual networks - allowed to reduce vanishing gradient problem while still being relatively deep network.
Original paper proposes multiple versions that differs with number of layers:
- ResNet18
- ResNet34
- ResNet50
- ResNet101
- ResNet152

First proposed solution is baseline model. It employs first vision of model ResNet architecture: **ResNet-18**. 
Since this model was primarily built to solve image classification problem, there are required adjustments to enable it to solve regression problem:
- classification head is replaced by dense layer with one numeric output.

Second solution employs bigger version of that architecture **ResNet-101**. 
To enable this model to solve regression problem the same architecture changes were applied as for baseline model. 

Both models were trained in multi-node SageMaker environment utilizing PyTorch framework that enables Distributed Data Processing:
- copies of model are stored on all nodes
- training and validation subsets are split across nodes 
- forward and backward are run phases independently for nodes
- weights are updated based on averaged gradient changes from all nodes

Training phase was performed on two SageMaker instances: `ml.g4dn.xlarge`. 
This is the cheapest instance that provides GPU capabilities. 
It offers **NVIDIA T4 GPU with 16 GB vRAM**, **4 vCPUs** and **16 GiB memory**

For training, pre-trained version of these ResNet variants were obtained from PyTorch library. 
Both models were pre-trained on the ImageNet-1K (ILSVRC 2012) dataset.
All weights of backbone part were frozen and only the last layer was trained.

During training phase, following hyperparameters were used:

| Hyperparameter                                    | Value |
|---------------------------------------------------|-------|
| Mini batch size                                   | 512   |
| Number of epochs                                  | 5     |
| Learning rate                                     | 0.003 |
| Early stopping after n epochs without improvement | 2     |

For inference to get the final predications, predicted values should be rounded to closest non-negative integer.

## Evaluation

In order to evaluate performance of these solutions, test subset with 80,596 images is utilized. 
For each of these solutions **AWS Sagemaker Transform Batch** Job is set up to calculate responses for all images from test subset.
Results of these computations allows to measure models' performances.

For evaluation purposes following metrics are utilized:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R squared (R^2)

These metrics are typical for regression problems.

Performed procedure allowed to obtain following results:

| Model      | MAE    | RMSE   | R^2    |
|------------|--------|--------|--------|
| ResNet-18  | 2.3564 | 5.4390 | 0.0909 |
| ResNet-101 | 2.2805 | 5.3981 | 0.1046 |

First metric (MSE) shows that obtained models make mistakes. The Mean Absolute Error for *ResNet18* is 2.36, while for *ResNet101* is 2.28.
On average predictions are off by about 2-3 items. While for large bins (above 100 items) this might be minor error, for small bins (1-10 items) it's quite significant.

Second metric (Root Mean Squared Error) reports 5.44 and 5.40 for *ResNet18* and *ResNet101* respectively. 
Because RMSE squares individual errors before averaging, it penalizes larger Kmistakes heavier than MSE.
This suggests that both models may struggle more with larger bins where bigger errors can occur.

The final metric indicates *ResNet18* explains only ~9.1% of test subset invariance, while *ResNet101* explains ~10.5% of test invariance.

All metrics consistently indicates that bigger model *ResNet101* performs slight better than smaller one *ResNet18*.

## Summary

As part of this project ResNet architecture was employed to solve object counting problem. 
Architecture was adjusted to support regression problem. Two variants (*ResNet-18* and *ResNet-101*) were fine-tuned.
Obtained results indicate that on average obtained models' predictions are off by 2-3 items.

In this project, *AWS SageMaker* services was utilized:
- AWS SageMaker Studio as web-based IDE
- S3 Bucket to store dataset
- AWS SageMaker Jobs to train 
- AWS SageMaker Batch Transform Job to retrieve responses for test subset for models' evaluation


## Literature

[1] *Amazon Bin Image Dataset* was accessed on *2026-11-17* from https://registry.opendata.aws/amazon-bin-imagery  

[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: Deep Residual Learning for Image Recognition, 
https://doi.org/10.48550/arXiv.1512.03385
