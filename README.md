# steer-prediction
Udacity's Self-Driving Car Nanodegree project 3 - Behavioural Cloning

In this project I use openCV to read image, which is in the format of BGR. 

However, the image read out from Udacity simulator is RGB iamge, thus to plot the image, I need to convert to BGR using openCV:
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


