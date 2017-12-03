# **Behavioral Cloning Project**

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have used keras native to Tensorflow 1.4.
I have based my model on the paper "End to End Learning for Self-Driving Cars", with some minor enhancements.

- I added an additional 1 x 1 conv layer on the top after cropping with 3 filters (model.py line 93). Intuition behind this is to let the model figure out the required color scheme to use rather than converting it to greyscale or RGB or HSV etc. This resulted in lower validation and test losses.
- I have use LeackyRELU instead of RELU activation for all the dense layers, this resulted in lower validation loss. (model.py lines 105-111). Intuition was to smoothen the values as LeackyRELU gives more a smooth activation. The steering angles were jumping huge values, creating jerky movements during the drive. LeackyRELU addressed the issue to some extent.

<pre>
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 65, 320, 3)        12        
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 348,231
Trainable params: 348,231
Non-trainable params: 0
_________________________________________________________________
</pre>


#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer. I tried various learning rates, but the default works the best in this case.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (2 laps) & driving the opposite direction (1 lap). I did not record data for the recovering from the sides.

For details about how I created the training data, see the next section. 

#### 5. Solution Design Approach

The overall strategy for deriving a model architecture was to get a smooth steering angle with each image.

My first model was to use a set of conv layers followed by dropouts and batch normalizations layers. The model loss was very high but it did converge fast and overfit (high loss on the validation set). The model did well on the roads where there were lanes, but not so much on the roads with no lanes (yellow or white markings).

My second model is based on the paper "End to End Learning for Self-Driving Cars", with some minor enhancements.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The final step was to run the simulator to see how well the car was driving around track one. During the turns, the car steering was little jerky. I used LeackyRELU instead of RELU. This resulted in smooth value angle predictions. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 6. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center lane driving](https://raw.githubusercontent.com/ShankHarinath/CarND-Behavioral-Cloning-P3/master/images/Actual.jpg)

I then recorded the vehicle by driving in the opposite direction for 1 lap. Here is an example image:

![Center lane driving](https://raw.githubusercontent.com/ShankHarinath/CarND-Behavioral-Cloning-P3/master/images/Reverse.jpg)

I did not record the vehicle recovering from the left side and right sides of the road back to center.

To augment the dataset, I randomly flipped images, used the left and the right camera with 0.25 as angle correction.

After the collection process, I had 13500 number of data points. I then preprocessed this data by 
- converting images from BGR to RGB
- normalizing the images
- cropping the image to only capture the road

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was ~20, I had used early stopping, which would stop training around 20 epochs based on the validation loss. I used an adam optimizer, tried custom learning rate, but default works best.