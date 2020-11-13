# Fix_Resnet50_gender_classification
* paper reference: Face and Gender Recognition System Based onConvolutional Neural networks

![image-20201111144240133](https://github.com/Kimyuhwanpeter/TF-2.3.0_Fix_resnet50_for_gender/blob/main/1.JPG)

# Implementation
* Ubuntu 18.04
* Python 3.7.8
* tensorflow 2.3.0

# Detail
* This version is modified resnet-50 training in this paper.
* 2-fold cross validiation (different from paper)
* Use test accuracy during training instead of validation accuracy
* Use tensorboard to show the loss and acc graphs
* First fold for (generated from Morph)AFAD acc is 76.4 %
* Second fold for (generated from Morph)AFAD acc is 83.04 %
* (Generated from Morph)AFAD acc is 79.72 %
