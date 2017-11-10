# Steer-prediction for Self Driving Car

Udacity's Self-Driving Car Nanodegree project 3 - Behavioural Cloning

The scope of project is to teach car about human driving behavior using deep learning so that the car can predict steering angle by itself. Data collection, driving and testing are performed on Udacity car simulator.

### Overview
The project is consisted of the following modules:
- Setup and Environment
- Explorting the data (data_visualization.ipynb)
- Data Processing (utility.py)
- Deep Learning Model Architecture Design (model.py)
- Model Architecture Characteristics
- Model Training (Include hyperparameter tuning) 
- Results / Driving (drive.py)
- Lessions Learned
- Future Work

## Installation & Resources
1. Anaconda Python 3.5
2. Anaconda Environment(https://anaconda.org/hl395/autodrive35/2017.11.01.1933/download/autodrive35.yml)
3. Udacity Car Simulator on [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
4. Udacity [sample data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

## Files and Usage
`model.py` : training codes.

`model.json`: saved training model.

`model.h5`: saved training weight.

`drive.py`: code that take in image from the simulation and run through `model.json` and `model.h5` to predict steering angle.

To use: start Simulator, pick track and choose Autonomous mode. On terminal, access to folder where the files are saved and type `python drive.py model.json`

### Quickstart
**1. Control of the car is by using button on PC keyboard or joystick or game controller.**

:arrow_up: accelerate :arrow_down: brake :arrow_left: steer left :arrow_right: steer right

**2. Two driving modes:**
- Training: For user to take control over the car
- Autonomous: For car to drive by itself

**3. Collecting data:**
User drives on track 1 and collects data by recording the driving experience by toggle ON/OFF the recorder. Data is saved as frame images and a driving log which shows the location of the images, steering angle, throttle, speed, etc. 
Another option is trying on Udacity data sample.

## Test Drive
Drive around the tracks several time to feel familiar with the roads and observe the environment around the track.

Track 1: *flat road, mostly straight driving, occasionally sharp turns, bright day light.*
![track1](https://cloud.githubusercontent.com/assets/23693651/22400792/a8927a68-e58c-11e6-8a66-839869832cce.png)

Track 2: *hilly road, many light turns and sharp turns, dark environment*
![track2](https://cloud.githubusercontent.com/assets/23693651/22400796/be938938-e58c-11e6-9938-6ba32ef3d554.png)

## Project Requirement
Deep Learning training is on **Track 1** data. To pass the project, the car has to successfully drive by itself without getting off the road on **Track 1**. 
For self evaluation, the model can successfully drive the entire **Track 2** without getting off the road.

## Approach
To have any idea to start this project, [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) by Nvidia is a great place to start.
From the paper, data collection is the first important part. Per project requirement, data collection can only performed on **Track 1**. I drove about 4 laps around **Track 1** by keyboard control to collect data. The driving wasn't extrememly smooth as actual driving. So I decided to use Udacity sample data as starting point.

In this project I use openCV to read image, which is in the format of BGR. 

However, the image read out from Udacity simulator is RGB iamge, thus to plot the image, I need to convert to BGR using openCV:
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

### Understanding Data
There are 3 cameras on the car which shows left, center and right images for each steering angle. 

![views_plot](https://cloud.githubusercontent.com/assets/23693651/22402134/546e68ec-e5ba-11e6-9266-ff9d7fdf3431.png)

After recording and save data, the simulator saves all the frame images in `IMG` folder and produces a driving_log.csv file which containts all the information needed for data preparation such as path to images folder, steering angle at each frame, throttle, brake and speed values.

![driving_log](https://cloud.githubusercontent.com/assets/23693651/22401702/65c154a6-e5ab-11e6-966f-c39d0f6aaa9c.png)

In this project, we only need to predict steering angle. So we will ignore throttle, brake and speed information.

### Training and Validation
Central images and steering angles are shuffle and split into 70/30 for Training/Validation using `shuffle` & `train_test_split` from `sklearn`

Training data is then divided into 3 lists, driving straight, driving left, driving right which are determined by thresholds of angle limit. Any angle > 0.15 is turning right, any angle < -0.15 is turning left, anything around 0 or near 0 is driving straight.

### Data Augmentation
* **Image Flipping**: In track 1, most of the turns are left turns, so I flipped images and angles (model.py line 19). As a result, the network would learn both left and right turns properly. Here is an image that has then been flipped:

![alt text][image5]


* **Brightness Changing**: In order to learn a more general model, I randomly changes the image's brightness in HSV space (model.py function *brightness_change*)

![alt text][image6]


**Data Balancing**

* **Collected data is not balanced**, we can see the steering angle historgram as shown below and data balancing is a crucial step for network to have good performance! 

![alt text][image7]

* In order to balance the data, we need to reduce the number of high bins, and I did it as in function *balance_data* in model.py. After the steps of collection, data augmentation and data balancing, I had 11120 number of data points. The results are shown below. 

![alt text][image8]

**Image Crop**

* In the image, the up part (sky) and bottom part (front part of the car) are not very useful for training, and on the other hand, it might lead to overfitting. So that I decided to crop out only the most useful part, and this is done in GPU for efficiency (model.py line 144) 

![alt text][image9]


When we process the left and right camera, we add corrections (+0.2 or -0.2) for their steering angles because we only know the ground-truth steering angle for the center camera (as given by Udacity simulator). Therefore, it may introduce some small errors for the steering angles of left and right images. So, I decided that in the validation data, I only use the center camera. Finally randomly shuffled the data set and put 30% of the data into a validation set (code line 214). 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 4 as evidenced by the validation loss is not getting lower anymore. I used an adam optimizer so that manually training the learning rate wasn't necessary.




### Recovery
In general sense, driving behavior can be trained using the central images because we always look ahead when driving. Driving is mostly straight driving as well or small adjustment to turn the car unless there is a sharp turn. Below is the plot of steering angle on track 1 from Udacity data.

![scatter](https://6f.png)
![distribution](https://a0.png)

But from inutition, if our car goes off lane (for example, distraction during text and drive), we can adjust it back on track right away. The machine doesn't have this intuition, so once it goes off road it would be hard to recover. To teach the machine this kind of recovery knowledge, we have to show it the scenarios. Hence, we use left and right camera images. Udacity gives out a great hint how to apply this method.
>In the simulator, you can weave all over the road and turn recording on and off. In a real car, however, that’s not really possible. At least not legally.
>So in a real car, we’ll have multiple cameras on the vehicle, and we’ll map recovery paths from each camera. **For example, if you train the model to associate a given image from the center camera with a left turn, then you could also train the model to associate the corresponding image from the left camera with a somewhat softer left turn. And you could train the model to associate the corresponding image from the right camera with an even harder left turn.**

So the task is to determine when the car is turning left or right, pick out a set of its left or right images and add/subtract with an adjustment angle for recovery. The chosen left/right images and adjusted angles are then added into driving left or driving right lists. Here is the logic:
  1. Left turn: + adjustment_angle on left image, - adjustment_angle on right image
  2. Right turn: + adjustment_angle on right image, - adjustment_angle on left image

### Preprocessing
1. To help the system avoid learning other part of the image but only the track, user crops out the sky and car deck parts in the image. Original image size (160x320), after cropping 60px on top and 20px on the bottom, and cropping 10px from left and right, the new image size is (80x300).
2. To help running a smaller training model, images are scaled to (200x66) size from cropped size (80x300).

### Generators
The model is trained using Keras with Tensorflow backend. My goal is to not generate extra data from what has been collected. To help always getting new training samples by apply random augmentation, fit_generator() is used to fit the training model.

There are two generators in this project. **Training generator** is to generate samples per batches to feed into fit_generator(). At each batch, random samples are picked, applied augmentation and preprocessing . So training samples feeding into model is always different. **Validation generator** is also to feed random samples in batches for validation, unlike training generator, only central images are used here and only proprocessing is applied.

### Training
After many trial and error in modify Nvidia model, below are my best working model.
- 1st layer: normalize input image to -0.5 to 0.5 range.
1. First phrase: 3 convolution layers are applied with 5x5 filter size but the depth increases at each layer such as 24, 36, 48. Then, 2 convolution layers are applied with 3x3 filter size and 64 depth. To avoid overfitting at convolution layers, Relu activation is applied after every convolution layers.
2. Second phrase: data from previous layer are flatten. Then dense to 80, 40, 16, 10 and 1. At each dense layer, 50% Dropout is also applied for the first 3 dense layer to avoid overfitting.
 With recommend from other students, L2 weight regularization is also applied in every convolution and dense layer to produce a smoother driving performance. After many trial and error, 0.001 produce best peformance for this model.
3. For optimizer, Adam optimizer is used. I started with 0.001 training rate but 0.0001 seems to produce a smoother ride. Therefore, I kept 0.0001 learning rate.

The final working weight was trained with 20 epoch, 0.27 adjustment angle and 64 batch size. To run training: `python model.py --epoch 20`

![architecture](https://cloud.githubusercontent.com/assets/23693651/22402330/ac793d4a-e5c0-11e6-9c41-a014fe3dd1a7.png)

![training2](https://cloud.githubusercontent.com/assets/23693651/22402343/f892ac92-e5c1-11e6-82da-ce39e51a96be.png)

### Testing
Use the training model that was saved in `model.json`, and weights in `model.h5`. Make sure the feeding images from test track is preprocessed as well to match with final training images shape in the training model.

To run test: `python drive.py model.json`

https://youtu.be/mR6Gswp5Xmo

![track1]()

https://youtu.be/hZfchwEIqqU

![track2]()

### Lessons Learned/Reflection

While the car keep running in the simulator, I can prepare the document for the submission. This is the self driving car future we are looking for. We need overcome some obstacles: 
- Better training data
- More compute power
- Standardize Network Architecture
- It is a try and error approach, and engineering approach
- Working on real environment

### Future Work
The Udacity provided training data is not too bad. It does have some bad turns. And the model clones them. It is perfecty matched the title "behavioral clone". If you are good driver, it clones. If you are bad driver, it clones too. 
I think somehow it is hard to figure out who is better driver, human or machine.  

In the begining, I don't like this kind of sharp on and off keyboard driving input. Later, I found out the udacity dataset is already smoothed, or at least is joystick or wheeled controller input. I thought a smooth steering curve is better for the training, but it turn out not really the case. Over smoothed curve yield very aggressive turnning. Maybe it is a very good result, we just need fine tune the controller to handle it properly. Also, in the real world, most of the time, the steering wheel is in netural position. Train the machine not to over react is harder than keep moving. 

190 seconds to train 19200 66x200 size images on CPU, is not that bad. Somehow I find the GPU is not working as hard as I expected in the generator setting. Both my GTX 1070 or K2000 GPU Utilization is very low, less than 4-30%. On normal tensorflow test, the GPU get at lease 3-10 times faster than CPU. I guess the bottle neck maybe is the generator. During training or driving, the RAM memory useage is less than 2.8G, It doesn't make sense to save memory but spend more time waiting for results. After tried different network architecture, Nvidia, Comma AI and my home made one, I like the Nvidia one. I hope they can standardize it, and provide with trained weights as well. Transfer learning has huge benefit for machine learning.

I find the simulator also provide real time steering_angle, throttle, speed and image feed. Therefore, it is possible to record new training set driving by the machine. Then train the machine again. After few generation, the machine driver will be better than human. I am going to explore more about the reinforcement learning. 

### Acknowledgements

There are many online resources available and helpful for this project. Thank you everyone to share them to the world. 
-  https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff#.kot5rcn4b
-  https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.dttvi1ki4
-  https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.ykemywxos
-  https://github.com/upul/behavioral_cloning
-  https://review.udacity.com/
-  http://stackoverflow.com/questions/1756096/understanding-generators-in-python
-  http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
-  The model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
-  Source:  https://arxiv.org/pdf/1604.07316.pdf
-  https://github.com/mvpcom/Udacity-CarND-Project-3
-  https://github.com/karolmajek/BehavioralCloning-CarSteering
-  https://github.com/commaai/research/blob/master/train_steering_model.py
-  https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
-  https://github.com/vxy10/P3-BehaviorCloning
-  https://github.com/ctsuu/Behavioral-Cloning
-  https://github.com/ancabilloni/SDC-P3-BehavioralCloning/
-  and many many comments in slack channels. 




