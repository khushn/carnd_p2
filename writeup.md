# **Traffic Sign Recognition** 
---


[//]: # (Image References)

[image1]: ./examples/traffic_sign_counts.png "Visualization of traffic sign numbers"
[image2]: ./examples/before_after_normalization.png "Normalization"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./images_from_web/sl_70.jpg "Traffic Sign 1"
[image5]: ./images_from_web/wild_animals.jpg "Traffic Sign 2"
[image6]: ./images_from_web/priority.jpg "Traffic Sign 3"
[image7]: ./images_from_web/sl_80.jpg "Traffic Sign 4"
[image8]: ./images_from_web/stop.jpg "Traffic Sign 5"
[image9]: ./examples/visualize_conv1.png "visualize conv1"
[image10]: ./examples/visualize_relu1.png "visualize relu1"
[image11]: ./examples/visualize_conv2.png "visualize conv2"
[image12]: ./examples/visualize_relu1.png "visualize relu2"
[image13]: ./images_from_web/children.jpg "Children crossing"

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

Here is an exploratory visualization of the data set. We Also show a random image, along with its label picked up from the dictionary of signs. 

![alt text][image1]

### Design and Test a Model Architecture

####1. Preprocessing using greyscale and normalization

Code: 4th code cell

I used both first greyscaling followed by normalization as a technique to preprocess. Because I wanted to bring the values in a range of 0 to 1 from 0 to 255. I used OpenCV's normalize function for the same. 

Here is an example of a traffic sign image before and after normalizing.

![alt text][image2]


#### 3.  Final model architecture

Code: 6th to 9th cell

My final model consisted of the following layers. Its basically a LeNet, but I removed the maxpool layers after each convolution, for reasons mentioned in 'The training of the CNN' below this section.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
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

I used AdamOptimizer, with no. of Epochs as 30. After training various times I noticed that accuracy was saturating after a while. Batch size of 250. On removing the two max pool layers (discussed below), the model size became huge. The model file size on disk is 182 MB. Whereas as with MaxPool layers it was just around an MB. That is because of the big fully connected layer fc1(9216x1200), which alone has 9 million weights. 


#### 5. Approach for finding the solution

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

#### Started with LeNet
I started by using the LeNet architecture. After some 8 tries with different hyper parameter values (mainly varying the learning rate from .001 to .0001) and changing the no. of epochs, the validation accuracy, remained just below .93. 

##### Removed Max pool layers
After that I removed the two max pool layers. Then validation accuracy jumped upto .949 (well above the required threshold). This definitely made the model much bigger. Even on a GPU it took around 10 minutes to train. But that involved a lot of tries of different models

My final model results were:
* Training set accuracy of: 1 (reached around 10th Epoch)
* validation set accuracy of: .949
* test set accuracy of: .932

Since the test accuracy, is below the validation accuracy, I believe the model does not suffer the problem of overfitting.

#### Below is the summary of various models tried by me, and insights:
 
| Epochs         		|Validation Accuracy| Model type | Sigma | Insight
|:-----------------:|:----------------:|:----------:|:-----:|:------:|
|100|  .86|CNN filter size=3x3 and Maxpool layers|sigma=.1|Small model size fast training|
|200| .901|same as above|sigma=.1|Model size not big enough to train|
|100|  .91|Made filter size 5x5|changed sigma=.2| Increasing the model size increase accuracy. More capacity!|
|200|.932| ""| ""| interim step|
|400|.927| ""| ""| Just wanted to try high # of Epochs. Validation accuracy came down.|
|50| .921| Remove Max pool later below second layer| ""| Accuracy improved, as model got bigger!|
|150| .922| ""| ""|And accuracy saturated around that|
|50| .902|Removed both max pool, but increased 2nd filter to 7x7|""| Model became huge, but no betterment in validation accuracy, perhaps because of bigger filter|
|50|.901|Reduced filter size to 5x5, Same as LeNet| ""| Still the same!|
|50|.951|Model same but changed sigma|changed sigma=0.1| The accuracy jumped when sigma was lowered. Key Insight! Its discussed below|
|30|.949|Same model, but retrain  upto Validation accuracy saturation| ""| Bingo!|

Below are some core insights regarding learning rate, sigma and use of color images. 

##### Learning rate was .001 for all the training done above: 
Few times I played with a small learning rate. But since all the values are small because of normalization. It was really ineffective. At the core its multiplication of numbers. And so learning late is relative to that. That was a core insight of so much training.

#### Sigma
I tried a lot of models with sigma = 0.2. But they saturated much below the desired level. I guess that's because of lot of noise in the weights, and it takes much longer for the gradient descent to work. It possibly even failed, by getting stuck in some local minima. So sigma=0.1 was just right. 

#### Use of color images
I tried out with color images 32x32x3 and used the LeNet (minus MaxPool) model, same as that described above. With that validation accuracy was .96 and test was .943. But I decided to retain the grey scale image one, as wanted to be cautious against overfitting.

### Test the Model on New Images

#### 1. German traffic signs found on the web 

Speed Limit 70: 
![alt text][image4] 

Wild animals crossing: 
![alt text][image5]

Priority:
![alt text][image6] 

Speed Limit 80: 
![alt text][image7]

Stop: 
![alt text][image8]

All the images were hard, as they are of different sizes. Only 5 are shown above but I got a some 20 or so images, and tried all of them. My accuracy was low. Varied between .545 to .385. My final accuracy, as I added new images. It was a humbling experience for me, as this was much below the test accuracy of .93+. Discussion and insights are below

####2. Discussion on the 'new images' results

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| Discussion|
|:---------------------:|:---------------------:|:---------:| 
| Speed Limit 70      		| Speed Limit 70| Fine!| 
| Wild animals crossing 	| Slippery Road			| In the probability distribution, wild animals crossing was second with a low probability, if that's some consoltation|
| Priority					| Priority											| Fine|
| 80 km/h	      		| 80(in one case); 30 in others	| 30 is similar to 80. Also I realized that my images were bigger than what are there in test DB. Have analyzed that separately, below|
| Stop			| Stop      							||


The model was able to correctly guess 3 and 1/2 of the 5 traffic signs, which gives an accuracy of 70% for these 5. 

More discussion below.

#### 3. Softmax probabilities

Code: is in the Ipython notebook in the cell below the prediction code cell.

|Sign| Top 5 probabilities|
|:---:|:--------------:|
|Speed limit 70|0.989675 Speed limit (70km/h)|
|-|0.0103248 Speed limit (20km/h)|
|-|1.28908e-08 Speed limit (30km/h)|
|-|2.38599e-14 Speed limit (80km/h)|
|-|1.79223e-14 Roundabout mandatory|
|Wild animals crossing|0.999626 Slippery road|
|-|0.000369339 Wild animals crossing|
|-|3.19461e-06 Road work|
|-|1.3045e-06 Double curve|
|-|1.28474e-08 Beware of ice/snow|
|Priority|0.997281 Priority road|
|-|0.00271914 Ahead only|
|-|1.59808e-08 End of all speed and passing limits|
|-|7.07516e-09 Speed limit (30km/h)|
|-|6.29958e-09 End of no passing|
|80 km/h|1.0 Speed limit (80km/h)|
|-|7.0043e-11 Speed limit (50km/h)|
|-|1.94096e-12 Wild animals crossing|
|-|2.45505e-13 No passing for vehicles over 3.5 metric tons|
|-|1.31589e-13 Speed limit (100km/h)|
|Stop|0.982638 Stop|
|-|0.0100575 Ahead only|
|-|0.00393836 Yield|
|-|0.00329273 Turn right ahead|
|-|5.53279e-05 No vehicles|


I decided to look at individual signs test accuracy, to understand individual signs better, below is one below. 

### Accuracy of prediction of 80 km/h speed limit on the test set
Code: Last second cell has the code

Results: total:  630  count of  Speed limit (80km/h)
Validation accuracy: .94 

### Visualize the Neural Network's State with Test Images
Code: Last section (end of the notebook)
Image used (Children crossing: 

![alt text][image13]

The code for this was slightly tricky, as I had already trained the model without the layers being named. But thankfully, after some forum help, was able to know and use sess.graph.get_tensor_by_name() function: 
e.g. conv1 = sess.graph.get_tensor_by_name('Conv2D:0')

Below are the visualization feature layers: 
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

#### Visualization analysis
Conv1 layer features have lot of similarity with the original image. But even there each cell highlights some different things. Example some have caught the right slanting edge, and others the left. The Relu layer makes the image more dark. Naturally so, as it catches only values above 0. (I guess matplotlib normalizes the values in an above 0 range, before plotting). So Relu layer shows only the most prominent features. By Conv2 layer number of feature matrices are 16. So I guess, each one looks for something specific. For some its hard to tell what they are looking. Again Relu2 after conv2, becomes even more sparse. Some where in the bits of that layer, we have encoded the necessary high level features. As the thing works!

