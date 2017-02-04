##**Project Approach**
Behavioural Cloning Project is to demonstrate the reuse a pre-trained model and its trained weights to use it in real-time situations to predict and thereby demonstrate the behaviour learned.  From the overall point of view, it is cloning the behaviour of humans by machines. 

As part of the project, a simulator is provided to generate the data and evaluate the model trained.  The solution of this project is arrived in a systematic manner as listed below.

- Download & use the simulator as per instructions
- Generate a sample data using simulator
- Write a code to read the csv file, load and visualize the images & labels (steering angle)
- Develop a simple model for sample and complete the skeleton code along with saving model & weights
- Run this code on the sample data developed and save the model & weights file
- Evaluate the trained model with simulator as per instruction
- Include a Python generator for feeding data to the model

Reaching this stage is a milestone of the project.  Now, start the real portion of the project by elaborating the working code to solve the given problem with following steps.  Train & evaluate the model at every step to achieve the desired behavior of simulated vehicle in the autonomous driving mode.

- Generate a reasonable size of input data using simulator
- Augment additional images by Flipping, adjusting brightness, transforming, etc.
- Tweak the model based on the evaluation outputs
- Repeat the above steps till the desired result is achieved.

##**Architecture**
Convolution Neural Networks are the best deep learning models to analyse and learn from images. The same has been proved by LeNet, AlexNet, GoogLeNet, ResNet, VGG, etc.  For the given problem at hand, started with a simple Convolution layer and then a Dense layer after flattening the data as a simple model architecture.  Those models failed miserably in the evaluation, but it was an expected outcome. It can be improved by strengthening the input training data and architecture of the NN model.  Let us talk about the model architecture in this section.  

Recollected the previous example architectures (like LeNet, AlexNet), and went through the NVIDIA model architecture as well from the link provided.  Adopted the NVIDIA architecture to use it with the current data, and trained the model and evaluated.  This time there was a significant improvement in the simulation. Then split the existing data into training and validation sets, and trained again.  In some of the epochs, the validation accuracy came to 1.000, which is an indication of over fitting the images.  So, added the drop out layers in the middle of architecture and able to train and evaluate again.  This time, there was some more improvement.  

In the same manner, changed the activation from Relu to ELU (Exponential Linear Unit), which is found to be superior than Relu. (Refer the article refered about ELU in the references).  Similarly, on the pooling layer also, tried Max and Average pooling.  And finally changed the optimizer to Adam optimzer, which is considered to be the best optimizer for image processing.  All these changed improved the output of the autonomous driving, and I was changing the data augmentation also in parallel while tweaking the model.

The final architecture arrived is described below. 

![Model Architecture](/Behavioural_Cloning_P3/model.png)

Figure-1: Model Architecture

#####**Layer-1:** Convolution layer-from 3 dimention to 24 dimension using 4x4 filter and 2x2 stride. Used ELU activation and Max Pooling with 3x3 filter

#####**Layer-2:** Convolution layer-from 24 dimention to 36 dimension using 3x3 filter and 1x1 stride. Used ELU activation, Dropout of 0.3 and Average Pooling with 2x2 filter

#####**Layer-3:** Convolution layer-from 36 dimention to 48 dimension using 4x4 filter and 1x1 stride. Used ELU activation, Dropout of 0.3 and Max Pooling with 2x2 filter

#####**Layer-4:** Convolution layer-from 48 dimention to 64 dimension using 2x2 filter and 1x1 stride. Used ELU activation, Dropout of 0.3 and Avg Pooling with 1x1 filter

#####**Layer-5:** Flatten the data from multi-dimension tuple to 1D tuple

#####**Layer-6:** Fully Connected Layer with the output size of 512, followed by a Dropout of 0.2 and ELU activation

#####**Layer-7:** Fully Connected Layer with the output size of 64, followed by ELU activation

#####**Layer-8:** Fully connected layer with output size of 1 as the final out layer.  Since it is a regression problem, no activation function is attached.

##**Dataset**
Generation of data using the simulator was a tricky part in this project.  As indicated in the NVIDIA paper referenced, there is a bias towards turning left and being straight with 0.0 steering ange. So, generating a data with a reasonable distribution of steering angle value was quite challenging.  On several attempts, tried to generate around 1500 to 3000+ samples.  However, they were not so good that could achieve the full cycle of autonomous driving in simulator.  After some time, decided to use the data provided by Udacity.  That data also was not having a good distribution of positive, negatize & zero steering angles.  The histogram of the data used is given below in Figure-2.

![Histogram of Input data](/Behavioural_Cloning_P3/original_histogram.png)

Figure-2: Histogram of Input data

In order to compensate the poor distribution, enormous amount of data needed to be augmented using pre-processing techniques. In my pre-processing,  additional images are generated by 

 - flipping images horizontally 
 - adjusting the brightness of the images
 - cropping the images to remove sky & vehicle bonnett - they are pretty much same and thereby can be called in-variants and not very useful features
 - resizing the image to 64x64
 - normalizing the images

Rotation & Affine Transformation were not done on the input images.  Since the position of image important, rotatio & transformation would confuse the model instead of supporting for this problem. Hence it was avoided. Sample plots of images taken at various stages are given below for reference.
![Original Images](/Behavioural_Cloning_P3/original_images.png)
Figure-3: Original Images

![Cropped & Resized Images](/Behavioural_Cloning_P3/cropped_images.png)
Figure-4: Cropped & Resized Images

![Brightness adjusted images](/Behavioural_Cloning_P3/brightness_adjusted_images.png)
Figure-5: Brightness adjusted images

![Normalized images](/Behavioural_Cloning_P3/normalized_images.png)
Figure-6: Normalized images that are fed to Model

##**Training**
As explained in the dataset generation and model architecture sections, training started with a minimal set of data and a simple model that could run on my laptop (Intel i7 2.5GHz 6500U processor with 8GB DDR3 RAM).  Then gradually increased the dataset size & layers in the architecture, soon my laptop started throwing memory errors.

As indicated in the project instructions, learned about the Python Generated and implemented it successfully.  To my surprise, it was able to handle Udacity data set training as well with out running into memory error. 

Once I got a reasonably performing model & dataset, the real tweaking started. Played around with following parameters to get the final solution.

 - Batch Size, Epoch size, Validation data size, Learning Rate
 - Optimizers, Activation Functions, Max/AvgPooling, Dropout percentage
   and even number of layers 
 - Dataset sizes, augmentations, number of
   samples per epoch, number of samples for validation, data
   crop/resize, etc.

##**Challenges**
This project provided a great opportunity to learn various aspects on our own by doing some quick search/research. Few challenges faced are:

 1. The simulator vehicle was turning left even in straight road.  It was very evident that the sample track was biased towards left turn, so used the other track data as well.  And added flipped images with the steering angle multiplied by -1 as part of augmentation.  It helped to keep the vehicle from turning left or right.
 2. Vehicle was vobbling even in straight road.  Applied normalization for the images and also for the autonomous simulator (in drive.py), then it reduced drastically.
 3. Validation data accuracy was 1.0, which is an indication of overfitting the data.  Introduced dropout layers and shuffling of both training and validation data.  Then the accuracy reduced a lot, but the model was performing better in the simulation.
 4. Long Processing Time.  Initially I was using the entire image of size 160x320, but half of the data in that image was not very useful features or in-variants. Example: Sky, Trees and Vehicle Bonnet was not helping a lot, so cropped the image to a region where the real useful features are available.  And also based on experts suggestions, resized the image to 64x64 which helped processing much faster without losing the accuracy of the results.
 5. Memory Problems: Initially I was not using the Python Generator, then created one and used it successfully to overcome the laptop capacity issue.  My current final model & data size doesn't even use the full available memory or processing capacity now to train the model.
 6. Model Architecture & Dataset: A lengthy trial & error exercise with some logical reasoning for each tweaks.  After two weeks of trial & error, able to get an optimal model that can successfully run the simulator vehicle in autonomous mode in both the tracks.

#####**References**

 - Augmentation by Prof.Vivek Yadav -
   https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.f0shx47rc
 - Behavioural Cloning Cheat Sheet -
   https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet
 - Oliver Cameroon's Step by Step guide -
   https://medium.com/udacity/coding-a-deep-neural-network-to-steer-a-car-step-by-step-c075a12108e2#.f6hg7fn94
 - Subodh's Blog -
   https://medium.com/@subodh.malgonde/teaching-a-car-to-mimic-your-driving-behaviour-c1f0ae543686#.7e2rokgdn
 - Sebastian Ruder's article on Optimization Algorithms -
   http://sebastianruder.com/optimizing-gradient-descent/ 
 - Blog on ELU -
   http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/
