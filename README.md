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
<br/>

* First fold for (generated from Morph)AFAD acc is 76.4 %
* Second fold for (generated from Morph)AFAD acc is 83.04 %
* (Generated from Morph)AFAD acc is 79.72 %
<br/>

* First fold for (generated from AFAD)Morph acc is 88.83 %
* Second fold for (generated from AFAD)Morph acc is 89.97 %
* (Generated from AFAD)Morph acc is 89.4 %
<br/>

* First fold for original AFAD acc is 97.2 %
* Second fold for original AFAD acc is 99.99 %
* Original AFAD acc is 98.59 %
<br/>

* First fold for original Morph acc 98.97 %
* Second fold for original Morph acc 99.99 %
* Original Morph acc is 99.48 %
<br/>

* First fold (Train-AFAD, Test-generated AFAD) acc 96.83 %
* Second fold (Train-AFAD, Test-generated AFAD) acc 99.98 %
* Train-AFAD, Test-generated AFAD acc is 98.41 %

* First fold (Train-Morph, Test-generated Morph) acc 99.3 %
