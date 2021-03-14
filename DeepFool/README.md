# DeepFool

Paper: DeepFool: a simple and accurate method to fool deep neural networks.

## Overview

DeepFool propose a method to fool the model by moving the data point to the decision boundary.

![](../Fig/Fig3.png)

By using an iterative attack, and linearizing after the decision function locally around the current point, we can sucessfully fool the network.

## How to use

Choose the index of image that you like, and run the ``deepfool.py``.

## Result

Result of DeepFool:

<img src="samples/sample1.png" style="width:50%"> clean image with sucessfull categorized as 7.
<img src="samples/ad_sample1.png" style="width:50%"> perturbed image, categorized as 9.

<img src="samples/sample2.png" style="width:50%"> clean image, sucessfully categoried as 2.
<img src="samples/ad_sample2.png" style="width:50%"> perturb image, categorized as 1.

<img src="samples/sample3.png" style="width:50%"> clean image, categorized as 0.

<img src="samples/ad_sample3.png" style="width:50%"> perturb image, categorized as 6.