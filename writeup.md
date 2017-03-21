# **Traffic Sign Recognition** 
---


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/before_after_normalization.png "Normalization"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./images_from_web/sl_70.jpg "Traffic Sign 1"
[image5]: ./images_from_web/children.jpg "Traffic Sign 2"
[image6]: ./images_from_web/no_entry.jpg "Traffic Sign 3"
[image7]: ./images_from_web/sl_80.jpg "Traffic Sign 4"
[image8]: ./images_from_web/stop.jpg "Traffic Sign 5"

## Rubric Points

Below I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  



### Link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1 The dataset
We the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) to train our convolutional neural network(CNN). It had been pre-processed by Udacity so that images are 32x32x3

Code: 2nd code cell of the IPython notebook.  

I used the basic python libs to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43


#### 2. Exploratory visualization of the dataset

Code: 3rd code cell  

Here is an exploratory visualization of the data set. We show a random image, along with its label picked up from the dictionary of signs. 

(TBD: It is a bar chart showing how the data ...)

![alt text][image1]

### Design and Test a Model Architecture

####1. Preprocessing using normalization

Code: 4th code cell

I used normalization as a technique to preprocess. Because I wanted to bring the values in a range of 0 to 1 from 0 to 255. I used OpenCV's normalize function for the same. 

Here is an example of a traffic sign image before and after normalizing.

![alt text][image2]


#### 3.  Final model architecture

Code: 6th to 9th cell

My final model consisted of the following layers. Its basically a LeNet, but I removed the maxpool layers after each convolution, for reasons mentioned in 'The training of the CNN' below this section.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x6     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5x16	    |  1x1 stride, valid padding, outputs 24x24x16.      									|
| RELU					|												|
| Fully connected(fc1)		| Input: 9216(24x24x16), output:1200        									|
| RELU					|												|
| Fully connected(fc2)		| Input: 1200 output:300        									|
| RELU					|												|
| Fully connected(fc3)		| Input: 300 output:43        									|

 


#### 4. The training of the CNN

Code: 10th cell

I trained the model on AWS GPU (g2.xlarge) instances. Starting point was the udacity-carnd AMI. In the end I built my own AMI, having the trained model, for possible future use.

I used AdamOptimizer, with no. of Epochs as 50. After training various times I noticed that accuracy was saturating after a while. Batch size of 128. On removing the two max pool layers (discussed below), the model size became huge. The model file size on disk is 137 MB. Whereas as with MaxPool layers it was less than an MB.


#### 5. Approach for finding the solution

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

#### Started with LeNet
I started by using the LeNet architecture. After some 8 tries with different hyper parameter values (mainly varying the learning rate from .001 to .0001) and changing the no. of epochs, the validation accuracy, remained just below .93. 

##### Removed Max pool layers
After that I removed the two max pool layers. Then validation accuracy jumped upto .96 (well above the required threshold). This definitely made the model much bigger. Even on a GPU it took around 10 minutes to train. 

My final model results were:
* validation set accuracy of: .96
* test set accuracy of: .944

Since the test accuracy, is below the validation accuracy, I believe the model does not suffer the problem of overfitting.
 

### Test the Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
