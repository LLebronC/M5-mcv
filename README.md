# Scene Understanding for Autonomous Vehicles with deep neural networks
Master in Computer Vision - M5 Visual recognition

## Project Documentation
### Project slides
- [Google slides](https://docs.google.com/presentation/d/1AHFAFCaj7uQkiXKEfW8QbdqtRUyyvBKobA6NQJt4pjs/edit?usp=sharing) [In process]

### Project description
The goal of this project is to study the use of deep learning to semantically segment images and extract some knowledge of the scene. To do it we will use different types of neural networks.

- [Project Paper](https://www.overleaf.com/read/zbhrkkjvwkjv) [In process]

## Group members
 - [Group 6:](https://github.com/LLebronC/mcv-m5)
  - Jose Luis Gómez (joseluis-master@hotmail.com)
  - Luís Lebron(luis.lebron@e-campus.uab.cat)
  - Axel Barroso (axel.barroso@e-campus.uab.cat)
  - Hassan Ahmed
  
<h2 id="WSum">Weekly summary</h2>

<p><a href="#Week 2">Week 2: Object Recognition</a></p>
<p><a href="#Week 4">Week 4: Object Detection</a></p>
<p><a href="#Week 6">Week 6: Object Segmentation</a></p>

<h1 id="Week 2">Week 2: Object Recognition</h1>

## Abstract
This second week goal is prepare an object recognition classifier using the state of art of Neural Networks. As a first step for the M5 project, we need to identify properly the elements in a image, how the project is focused in Scene Understanding for Autonomous Driving, the elements to identify will be the relevance for this function, how they can be cars, pedestrians, traffic signs, etc. In order to train and validate our classifier we use Datasets provided with related traffic elements, this Datasets are TSingHua-TenCent 100K, INRIA pedestrians, Belgium traffic signs, Daimler pedestrians and KITTI. The classifier will be based on Convolutional Neural network, studying the actual state of art for image classification, training the best models of this last years and comparing the results.

## Framework composition

A framework in Python is provided to prepare easily the models, read automatically the provided Dataset, splitting in Train, Validation and Test, where with a configuration file we can manage all the parameters needed to train the CNN. The framework use Keras library and supports Theano and TensorFlow, making easy the compatibility of both libraries and the creation or integration of models.

### Modules used

The following modules are being used for this week to understand properly the code and fix minor problems:
- Main file train.py: where we can see the calls of the different modules and the arguments to use when we run it.
- Configuration: module where a configuration file is read in order to automatize the preparation of the experiments.
- Models: model factory implementation to add easily new models to the framework
- Tools: where we can find the optimizer factory to use the different optimizers integrated to use in the CNN, also the dataset loader and related files.
- Callbacks: callback factory to personalize the callback to use when a experiment is running.

### Run an experiment

To run an experiment using TensorFlow we use the next command with optional parameters as input in []: CUDA_VISIBLE_DEVICES=1 python test.py -c config/TT100k_classfier.py -e ExperimentName [-s SharedPath (default:/data)] [-l localPath (defaut: /datatmp).
We can also define by default this input parameters in the train.py in the lines 81 to 88. This command read the Dataset from the shared path specified, make a local copy on the local path specified, read the configuration file and names the experiment.
Automatically the execution creates a Experiments folder on the local path and stores information relevant about this experiment, like weights of the model, results, log files, etc. The configuration file specified in the command is the main tool to configure and parametrize the different experiments to do, selecting the CNN, Dataset, training configuration and data preparation.

### Integrate a new model

The steps to add a new model to the framework are:
- Define the model in Keras language and store it in /models.
- Go to the model_factory.py file import your model or main function that creates your model (ex: from models.resnet import ResnetBuilder), add the name of the model that you want to use in the configuration file to call it in the make function, add the name inside the if of the line 82.
- Define the call of your model, adding in the function make_one_net_model the entry needed, follow the other nets already implemented as example.

## Code changes and additions

- Added the densenet model using the implementation from: https://github.com/titu1994/DenseNet, now the denset net 40 is available. For using this model, please set densenet into the config file (model_name).
- Small change. Modified TensorFlow download link in the code/README.m file, from https://github.com/Theano/Theano to https://www.tensorflow.org/install/install_linux.
- Modified Optimizer_factory from Tools folder, added learning rate parameter to the Adam and SGD optimizers in order to be useful.
- Added new CNN models for classification: Resnet50 and InceptionV3. Using the Keras implementation are available in the configuration file in the parameter "model_name".
- Support for Weight decay integrated for this Models: VGG16, VGG19, Resnet50 and InceptionV3. The Weight Decay use L2 normalization and is activated when the value of the parameter "weight_decay" of the configuration file is higher than 0.
- Added the different resnet models using the implementation from: https://github.com/raghakot/keras-resnet, now the resnet 18, 34, 50, 101 and 152 are available, using resnetXX in the config file, the resnet50 from Keras is called using resnet50Keras.
- Added support for transfer learning from different datasets (different number of class) and different path for pretrain weights and test weights.

## Configuration Hints

- Densenet is really heavy, it goes out of memory very easily in our GPU. The minimum size is 32x32, 64x64 images work fine. 40 is the default depth of the net, this parameter can be changed into model_factory.py as an input parameter for the method build_densenet(..., depth = 'x').
- Resize dataset: remember that Resnet50 and inceptionV3 need a resize for the dataset in order to process the input image properly and work. Minimum input size [139,139] for InceptionV3 and [199,199] for Resnet50. When the dataset is smaller than this size like TT100K, use the resize parameter close to the minimum in order to penalize minimally the computation time for epoch only in the models with this requirements.
- TensorFlow try to allocate memory on the GPU even if don't fit firstly, if you use batch sizes inappropriate or resize the image to higher dimensions you can experiment a drastical increment in the epoch calculation time, so it's important to take into account.
- Use weight decay with lower values (exponent -4 or -5), high values penalize the weight and allows less variations needing more time to train and higher loss values, still the CNN will learn, but not appropriately.

## Task summary
### Task (a): Run the provided code
- Analyze the dataset.
- Calculate the accuracy on train and test sets.
- Evaluate different techniques in the configuration file:
 -- Use different pre-processings.
- Transfer learning to another dataset (BTS).

### Task (b): Train the network on a different dataset
- Set-up a new experiment file to discriminate among pedestrians, vehicles, and cyclists on KITTI dataset.
- VGG16 model.
- Train from scratch or fine-tuning?

### Tasks (c) and (d): Implement a new network
- InceptionV3 and Resnet are already implemented in Keras. Other models will be positively valued.
- Integrate the new model into the framework.
- Evaluate the new model on TT100K.
- Compare fine-tuning vs. training from scratch.

### Task (e): Boost the performance of your network
- meta-parameters tuning
- data augmentation

## Tests Realized

### Weight Normalization
Tested the Weight normalization behaviour in the models trying a weight decay value of 0.1, 0.01 and 0.001 in VGG16.
- 0.1: High starting loss, around 150, each epoch reduce this loss constantly but the training step is really slow per epoch, reducing the loss in the validation data but poor changes in the accuracy score. Too slow and higher penalization for weight is not useful.
- 0.01: Similar to the previous case, but the lost start at 20 and drops slowly, being better but still too slow to be really useful.
- 0.001 and above: train fast, similar loss speed like without weight decay but with a better control of the train step and normalizing the weights.

### Train from Scratch VS ImageNet
In general, with sufficient time to execute, an appropriate Dataset size, both method should achieve similar accuracy if you choose a proper optimizer configurations. But sometimes isn't like this, the main difference is the starting point, from scratch the start loss are high and falls more slowly than use the ImageNet weight where the starting loss is lower. Also it's easy to find a plateau that makes really difficult to improve the model, having to try some other configurations or need more data. It's recommended to start with imageNet weight if the model support it, because you will guarantee a good weight start point helping to avoid plateaus or other problems with the loss, also you will save in execution time and can be useful to perform Fine Tune adding new layers of modifying the top of a model.

#### Fine Tune example

A Fine Tune example can be seen in the test Datasets section bellow, the experiment is with a ResNet 50 of Keras tested in TT100K Dataset. A first train step with the weight preload from ImageNet and frozen to keep the values during 19 epochs, where we can see how is slowly adapted due the learning of the top of the model part, the flatten layer and softmax. When is observed that the model can't adapt more the loss, we stop the training and restart again unfrozen the layer from the base model part, allowing the learning. From this point (epoch 19) we can see how the loss drops until the saturation of the train.
If we compare with the ResNet 50 from scratch just below this one, there aren't notables differences at the end, ResNet learn quite good the Dataset with the proper learning rate and data augmentation, don't needing the ImageNet weights, but the fact that we used this weights improve slightly the final score in validation due the differences in weight adaptation during the train phase, end with this little variation that benefits the Fine Tune case.

### Analysing different pre-processings

### Transfer learning to another dataset
We use transfer learning using VGG16 from the TT100K dataset to the BelgiumTSC dataset. [Table](https://docs.google.com/spreadsheets/d/1edlG_QVyAuVgNCifR7pAlTNp4yfkvcW-LEhC4yp5AbE/edit?usp=sharing)

- train VGG16 in the BelgiumTSC from scratch

![VGG16 Scratch plot](figures/plotVGG16_BTS_scratch.png?raw=true "VGG16 BelgiumTSC Scratch Experiment")

- train VGG16 in the BelgiumTSC with pretrain weights

![VGG16 Pretrain plot](figures/plotVGG16_BTS_pretrain.png?raw=true "VGG16 BelgiumTSC Pretrain Experiment")

### Analysing data augmentation

To use properly the data augmentation is important to know which information are we obtaining from a dataset and also the model type of our CNN. For example, if we try to identify traffic signs a low rotations, shift and affinities with the proper values can have sense to make our dataset more complete and our model more robust, but some parameter like flips can destroy the meaning of the sign or element to classify.
In general, all the experiments are performed with Data augmentation, but sometimes we try to don't use it to see if the network learns better or for refine finals epochs to adjust the model to the data. The data augmentation used normally are 10 degrees of rotation and a factor of 0,1 or 0,2 in shifts, zooms and shears.

### Practical differences between Adam and RMSProp

Analyse the differences between both method practically is not easy and can be about personal likes, but in your case we chose adam from RMSprop, because adapt better the learning rates, a generic value of 1-E04 or 1-E05 works fine for all the models normally, but RMSprop need to be adjusted better, having cases that you need a learning rate more low or somethings higher. We won't say that adam is better, but in our case makes more easy the learning phase and become a preference for us.

### Techniques to improve the network results

In order to improve the results of the network or train more fast, focusing in the evaluation loss and scores there are some hints learnt while we make the experiment that can be useful.
- Use data augmentation accordingly to the necessities of the dataset to increase the generalization power of your model, this normally add loss in the train, making a bit more difficult to adapt the model, needing more epoch to adjust and being beneficial for the validation.
- Data preprocessing is essential, but in our case a rescale of the data is the best option to score the best values in validation, if you use mean or std subtraction this performance will drop slightly.
- If you use a model from Keras compatible with ImageNet weight, the best option is preload this weight and make find tune or simply use it from the beginning, you will obtain less starting loss and a fast learning for your model.
- when your model don't learn more, try to repeat the experiment in our last checkpoint using a lower learning rate value, sometimes you can find a plateau in the gradient descent that can be solved using a low learning rate.

### Dataset Experiments

#### TSingHua-TenCent 100K (TT100K)

This Dataset can be find here: http://cg.cs.tsinghua.edu.cn/traffic-sign/tutorial.html It's a Dataset of traffic signs of 64x64, that consist in 16527 images for train, 1655 for validation and 8190 for test, being the most challenging one for us with 220 classes of signals to identify and for this reaon is the preferent Dataset to test our Models.

Ranking best result of the models on TT100K daset: https://drive.google.com/open?id=1DPU7IIxMk3xZAEAY_5cH4X0YRmexPwH_1sK1yAKgDQs
Inside the document we can find the weight and configuration files associated to each experimented and can be recreated in this code.

![CNN Ranking](figures/Ranking.png?raw=true "CNN Ranking Experiment")

If we analyze the results obtained, in general we can see that the test Dataset for the TT100K should have a really similar properties like the train, because for all models the accuracy obtained is really high. In other hand, we can pay attention to the loss rating where the differences are noticed and also the validation accuracy. For this case we can see how the ResNet 50 Keras is the best model, the lowest loss and highest accuracy for validation. We should say that this experiments aren't homogeneous, someone has more data augmentation, others ones don't have mean and std subtraction, ImageNet weight preload, etc. All this factors makes little differences, but in general this are the top values obtained for each experiment and model.

##### VGG16 Keras
![VGG16 plot](figures/plotVGG16.png?raw=true "VGG16 Keras Experiment")

The main configuration for this test was using preload weight for ImageNet, optimizer Adam with 1-E05 of learning rate, weight decay of 1-E04, rescaling, std and mean subtraction without Data augmentation. The graphic shows the evolution of the model until the early stops callback end the training step. The models starts with a low loss due the ImageNet weights.

##### VGG19 Keras
![VGG19 plot](figures/plotVGG19.png?raw=true "VGG19 Keras Experiment")

For this experiment the configuration are preload weight from ImageNet, optimizer adam with 1-E04 of learning rate and 0,001 of weight decay, only a rescale to the input data and a data augmentation with 10 degrees of rotation and shifts, shear and zoom of 0.1. We can see that the lowest loss for validation was between epochs 40-50, but the early stopping was defined using the validation accuracy in this case, scoring the best near the epoch 91. This example is similar to the previous one with VGG16, score a little more validation due the generalization of the data augmentation.

A common property of the VGG models is the amount of parameters that they have, needing a lot of memory and having heavy weight files. In general they allow small input images and are fast due the few layers in comparison to the other models. But the validation loss shows that this model is less stable and generalize a little worst than Resnet for example.

##### ResNet 50 Keras

For this example we should two experiments involving the ResNet 50 from the Keras implementation, one with Fine Tune and another one from scratch. In general this ResNet model is really smooth learning, don't have a high variation within epochs in validation like VGG, even when the loss is a little higher, the scores in accuracy are close. The main problem is that they need a minimum input size of 197x197, needing to resize the Dataset input and increasing the computation time for epoch.

- with Fine Tune:

![ResNet 50 Keras plot](figures/plotResnet50Keras.png?raw=true "ResNet 50 Keras Experiment")

For the find tune configuration we use a first step frozen the weight of the base model, preloading the ImageNet, we resize input image to 197x197, and use optimizer adam with 1-E05 of learning rate and 1-E04 of weight decay. Rescaling, mean and std subtraction from the input preprocessing, with data augmentation with 10 degrees of rotation, and 0.2 factor for shifts, shear and zoom.
The second step have the same configuration but with the weight unfrozen.

- From Scratch VS ImageNet preloaded weight:

![ResNet 50 Keras plot](figures/resnet50-ScratchVSImNet.png?raw=true "ResNet 50 Keras Experiment")

##### ResNet 152

This experiment using a new implementation of the resnet model, allowing the 152 layer model shows how the model learn from scratch and have a fast learning process like the resnet from keras, being also smooth. The train model is adapted really fast.

![ResNet 152 plot](figures/plotResnet152.png?raw=true "ResNet 152 Experiment")

The configuration for this experiment was resize of the data to 197x197, optimizer adam with 1-E05 of learning rate and 1-E04 of weight decay, rescaling, mean and std subtraction and data augmentation of 10 degrees of rotation and 0.2 factor for shifts, shear and zoom.

##### InceptionV3 Keras

The Inception V3 model from Keras using the weight from ImageNet learns really fast, we can see that in 2 epochs the train accuracy saturates, also it is not so smooth than the ResNet, but it's stable. Apart of this, the learn progression is good and the scores similar to the other ones.
We need to remark that the standard InceptionV3 needs a image size input of 300x300, due the final AvgPooling layer in the top of the model, in order to avoid this high resize of the images, we modified the Inception to allow lower resolution, modifying the padding and stride inside this AvgPooling layer to 4, with images lower than this 300x300. With this modification we allow a minimum of 200x200 input images, close to the resnet restriction in order to compare and don't have significant differences.

![InceptionV3 plot](figures/plotInceptionV3.png?raw=true "InceptionV3 Keras Experiment")

The configuration for this experiment was resize of the data to 200x200, optimizer adam with 1-E04 of learning rate and 1-E04 of weight decay, rescaling, mean and std substraction and without data augmentation.

##### DenseNet 40
![DenseNet plot](figures/plotDenseNet.png?raw=true "DenseNet 40 Experiment")

#### INRIA pedestrians

INRIA pedestrian is a dataset that contains pedestrian images cuts with size of 128x64, to train we have 13429 images and 5100 of validation. The Dataset can be found here: http://pascal.inrialpes.fr/data/human/.
The two experiment performed show how easy is to train and obtain good results in a binary classification, needing few epochs to achieve a really high accuracy to detect pedestrians.

##### InceptionV3 Keras
![InceptionV3 plot](figures/plotInceptionINRIA.png?raw=true "InceptionV3 Keras Experiment")

##### VGG19 Keras
![VGG19 plot](figures/plotVGG19INRIA.png?raw=true "VGG19 Keras Experiment")

#### Belgium traffic signs (TBSC)

Dataset of traffic signs like TT100K, but with less classes and images, 61 classes with 4575 images for train, 2520 for validation and test.
This dataset is more easy to learn, giving easily very good results for test with a resnet 50.

##### ResNet 50 Keras
![ResNet 50 Keras plot](figures/plotResnet50BTSC.png?raw=true "ResNet 50 Keras Experiment")

#### KITTI

he KITTI dataset provided is composed by 60060 train images and 14876 for validation, don't have test. The image shape is 128x128 and the classes are different types of vehicles, cyclists and pedestrians.

##### ResNet 50 Keras
![ResNet 50 Keras plot](figures/plotResnet50KITTI.png?raw=true "ResNet 50 Keras Experiment")

##### VGG19 Keras
![VGG19 Keras plot](figures/VGG19kitti.png?raw=true "VGG19 Keras Experiment")

<p align="right"><a href="#WSum">Back to summary</a></p>

<h1 id="Week 4">Week 4: Object Detection</h1>

## Abstract

The aim of this week 3 and 4 is use the CNN to do object detection. Using an input image we want to detect the objects inside this one, define a bounding box for each one with the class that they belong. Using the state of art in object detection we will focus in YOLOv2 and SSD neural networks. First integrating in your framework with the modification needed to make them work, finally we analyze and test their performance and behaviour in the TT100K and Udacity datasets.

## Detection Models

### You Only Look Once (YOLO9000)

One of the models that we used is the YOLO9000 know as YOLO v2. YOLO is a network used to object detection, where the idea is use a single neural network to divide the image input in square regions, then this squares region defines which class can be (confidence) and then the union of this region determine the size of the bounding boxes. The version 2 adds few tricks that improve the train and performance of the network like Batch normalization, Convolution with Anchor Boxes, dimension clusters and more.
The final output for each image input of 320x320, consist in a 10x10 region map of the image with 5 prior (bounding boxes sizes) and the vector that contains the bounding box coordinates plus the classes confidence.

### Single Shot MultiBox Detector (SSD)

A different approach from YOLO is the SSD model. Based on a feed-forward convolutional network, the SSD produce a colletion of bounding boxes and scores for the presence of object class instances in those boxes. Use a early network for image classification, followed by a multi-scale feature map using convolutions to predict detections, in each convolution is performed a 3x3xP kernel that produce a score for each category.
The final output for each image input of 300x300, consist in a detection with 8732 bounding boxes per class, that it’s reduced to one per class using non_maximum supression.

## Code changes and additions
- Added the SSD model implementation from https://github.com/rykov8/ssd_keras, using the prior file and weight shared there. We needed to adapt the GT annotations to the input format of the SSD to calculate properly the loss.
- Fixed the prediction functionality in configuration, now it works with detection models, YOLO, tiny-YOLO and SSD, the prediction is showed in the image set in a folder inside the experiment.
- Added and adapted all the utilities function of SSD to be useful in the framework.

## Problems found

During the realization of this week 3 and 4, we had problems that make us waste a lot of time finding a solution and debugging the code, because is quite difficult to implement new networks and adapt to the Datasets and metrics.
The YOLO model was provided ready to use, with metrics and loss functions, where the files involved were basically yolo.py and yolo_utils.py. But in order to add new metrics like F1-score or drawing the bounding boxes in an image output, it becomes difficult to implement because the code is not commented and you need to know properly the output of the network. In your case, we tried to implement F1-score using the metrics in yolo_utils.py but without success, we end using the script provided to test metrics.

Other part that was difficult was the drawing bounding boxes in images functionality. To implement this part, we adapted the prediction functionality in the configuration file and when the type is “detection” we use this prediction and save the results with the bounding boxes drawn. First, we needed to implement a generator queuer that iterates the batches and allow us to use this data and make predictions with our model. A second step of this process is after predict a batch of data, we need to process this network output to a final output where we have a bounding box per class and it’s drawable in a image. Fortunately this function are given in the SSD and YOLO models, you only need to adapt the number of classes and the bounding boxes annotation from the ground truth. The adaptation of the ground truth is a critical step, because each model is ready to be used with a certain bounding box and class definition vector, so we need to adapt the ground truth if we want to evaluate properly the model and calculate the loss. Even it doesn’t sound complicate, the reality is that adapt properly the ground truth is troublesome because you need to ensure that all work fine and the adaptation it’s perfect if you want that you model learn. A final step in the drawing function is use the a threshold to select with bounding boxes confidence for each class you select and you need to use non-maximum suppression to unify the bounding boxes of the prediction and have a final bounding box to draw in a image. At test, the drawing function work fine with YOLO but we have problems with the sizes in the SSD model.

In other hand, the SSD implementation was difficult, because you have the model in github ready to use in a form that don’t fit properly with the framework and you need to understand the code provided there and the ouw code to see where to integrate the models and all the functionalities that affects to this new model. For example, in data_loader.py where the GT is adapted to YOLO, you need to create a new adaptation to SSD, or our prediction implementation, now need to have into account the peculiarities of the output of the SSD. Also, prepare a ss_utils, with the loss function of this model, call it properly in the model_factory and so on.

All this problems, practically solved are to reflect that we needed to work a lot with the framework and we modified and adapted the code to solve them.

## Task summary
##### Task (a): Run the provided code
Use the preconfigured experiment file (tt100k_detection.py) to detect traffic signs with the YOLOv2 model.
- Analyze the dataset.
- Calculate the f-score and FPS on train, val and test sets.
- Evaluate different network architectures:
 -- YOLO
 -- Tiny-YOLO
##### Task (b): Read two papers
- You Only Look at Once (YOLO)
- Single Shot Multi-Box Detector (SSD)
##### Task (c): Train the network on a different dataset
- Set-up a new experiment file to detect among cars, pedestrians, and trucks on Udacity dataset.
- Use the YOLOv2 model as before, but increment the number of epochs to 40.
- Analyze the problems of the dataset as it is. Propose (and implement) solutions.
##### Task (d): Implement a new network
- We provide you a link to a Keras implementation of SSD. Other models will be highly valued.
- Integrate the new model into the framework.
- Evaluate the new model on TT100K and Udacity.
##### Task (e): Boost the performance of your network
One of the main problems with YOLO is that the net cannot find small object in images. That is why, we implemented a novel method based on Tiny-YOLO. We thought that upscaling the input image we could detect better the small objects. Thus, what we do is to take the input image (input shape: (320,320,3)) and create two branches. The first branch will do a convolution (output shape: (320,320, 16)) as it is done in Tiny-YOLO. The second one is our contribution, we do an upsampling of the image(output shape: (640,640, 3)) and do the convolution (output shape: (640,640, 16)), once we have done the convolution we do a maxpool (output shape: (320,320, 16)). It is then, when we take this two branches and we merge them. After this merge layer (output shape: (320,320, 32)), the structure is the same that it was before in Tiny-YOLO. We called the new net Tiny-YOLT, You Only Look Twice. 
By doing that we want to solve the problem of missing detections for small objects. For running the experiment the config/Udacity_detection_YOLT.py is ready. We do not train the net, since it could take days or even a week. What we have seen is that the net is learning. We have try to build this net as a proof of concept. This work was based on the paper 'Locally Scale-Invariant Convolutional Neural Networks' (https://arxiv.org/pdf/1412.5104.pdf). We ended up with a much easier version of that. 


## Tests Realized

The experiment realized uses a threshold of 0,6 for the confidence in each class and the optimizer adam with 1E-05 os learning rate.
We used a new dataset called Udacity, that have 3 classes (pedestrians, cars and trucks) that it's already prepared like TT100K to do detection.

### YOLO Experiments
##### TT100k_detection 10 epochs using YOLO [[Weights](https://drive.google.com/open?id=0B_RS7KGCOO8RUkx4VldoTVJ4bmc)]
In this experiment we use the basic configuration and see how it work from the start. First thing to notice is that the validation set works worst if we compare it with the results obtained in test: avg_recall 0.96 and avg_iou 0.727, scoring only a iou of 0.56, this happends with all the test done in TTK100_detection. Then we analized the cause of this difference looking the validation images and comparing with train, in general all the pictures selected in validation are really blurred being really challenger for the model, also there are few classes missed in the train and other minor differences, instead the test is alike train being normal to score this high results.

![YOLO_e_10 plotttk](figures/TTK100_yolo_10.png?raw=true "YOLO 10 epoch")

In the graphic we can see how the network learns fast, with few epochs also becasue the dataset is huge if we compare to other that we used.
In train the recall basically reach the maximum and IOU and in test we achieve this result due to the similarity in the images.
We can see below two samples of the output results, where they identified properly the traffic sign and the center, but still need to improve the adjustment of the bounding box.

![YOLO sample 1](figures/YOLO_sample1.png?raw=true "YOLO sample 1")
![YOLO sample 2](figures/YOLO_sample2.png?raw=true "YOLO sample 2")

Other metric to compare with the nexts experiments are. fps: 19.81 and F1 score about 0.63.

##### TT100k_detection 20 epochs using YOLO [[Weights](https://drive.google.com/open?id=0B_RS7KGCOO8Rajk4cmNac3JBaDA)]
The next experiment was if 10 epochs was enough to get a good result, so we increase the number of epoch to 20. 
Forggeting validation, this experiment score: 0.60 iou , 20.1 fps, 0.44 recall, and around 0.30 of f1. If we see the graphics we can see that needs more epoch to converge but the time need to do it makes impossible to test this theory.

![YOLO_e_20 plotttk](figures/TTK100_yolo_20.png?raw=true "YOLO 20 epoch")

##### TT100k_detection 10 epochs using tiny-YOLO [[Weights](https://drive.google.com/open?id=0B_RS7KGCOO8RaXJGX08zUGRwcmc)]
Other experiment using TT100k_detection was to use the tiny-YOLO and see who it perform. In test the scores was: 32.11 fps, 0.63 iou, 0.82 recall and 0.39 f1. This models is the faster and the lighter so it gains stability faster but it's also make very dificult to adapt to the problem so the measure of f1 is bad compare to the original experiment.
figures/TTK100_tiny_yolo.png

![YOLO_tiny plotttk](figures/TTK100_tiny_yolo.png?raw=true "tiny-YOLO 10 epoch")

The samples below shows the differences with the tiny YOLO respect to the YOLO, where we can see the bounding boxes aren't adapted yet in the same samples, with the same number of epochs, due the 

![Tiny-YOLO sample 1](figures/TNYOLO_sample1.png?raw=true "Tiny-YOLO sample 1")
![Tiny-YOLO sample 2](figures/TNYOLO_sample2.png?raw=true "Tiny-YOLO sample 2")

##### Udacity 40 epochs using YOLO [[Weights](https://drive.google.com/open?id=0B_RS7KGCOO8RR2FlYUFIR3J2eFk)]
The nexts experiments were using the Udacity dataset. As in the previous dataset we will take as a reference the YOLO model result.

In test it score: 19.8 fps, 0.69 recall, 0.57 iou and around 0.36 of f1. The result obtained here, if we compare with the images below are rare, because the images shows in general a proper detection of bounding boxes in the images, but the scores are bad, so probably there are some errors in the metrics.

![YOLO plot](figures/YOLOUgraphic40epochs.png?raw=true "YOLO Experiment")

Image samples that show a quite good detection if we compare with the scores obtained.

![YOLO sample 1](figures/YOLOU_sample1.png?raw=true "YOLO sample 1")
![YOLO sample 2](figures/YOLOU_sample2.png?raw=true "YOLO sample 2")
![YOLO sample 3](figures/YOLOU_sample3.png?raw=true "YOLO sample 3")

##### Trying data augmentation Udacity 40 epochs YOLO
Unfortunately, data augmentation don’t improve the results and even decrease slightly the results obtained, because the system is sensible to the bounding box ground truth, even if we modify it with the transformation on the image, we introduce problems, like bounding box at borders of the image with shifts, or if we try to rotate or warp the bounding box can be modified wrongly.

![YOLO plot](figures/YOLOUgraphic40epochs.png?raw=true "YOLO Experiment")

##### Udacity 40 epochs using tiny-YOLO [[Weights](https://drive.google.com/open?id=0B_RS7KGCOO8RVDZSMW5UWWU3MTQ)]
In this new dataset we also train the tiny-YOLO. As in the previous set of experiments it scores the faster result with 31.31 fps but the other scores are worse. It gets a recall of 0.28, a iou of 0.4 and a f1 of 0.36.

![tiny-YOLO plot](figures/TinyUGraphic.png?raw=true "tiny-YOLO Experiment")

### SSD Experiments

##### TT100k 20 epochs [[Weights](https://drive.google.com/open?id=0B6eUlGGeZ9wAa21VTVMzT0pGSEk)]

In this test, SSD in TT100K scores a really good results, higher than YOLOv2. We think that the approach of SSD for this dataset is better, because the traffic sign are really small and the SSD generate more windows at differents resolutions, being more multi-scale than YOLO. Also in test the frame rate is really high, beating again YOLO in this aspect also.

Avg Precission = 0.945439038396
Avg Recall     = 0.788267481027
Avg F-score    = 0.859728957481
Average FPS: 98.67

![SSD plot](figures/SSDTT100kgraphic.png?raw=true "SSD Experiment")

Seems that there is some problems to our code when we try to plot the Bounding Boxes, because the results using the script are really good, but the image plotted detects signals but don't fit properly the bounding box around it. We can see the bounding boxes problems below.

![SSD sample 1](figures/SSDTT_sample1.png?raw=true "SSD sample 1")
![SSD sample 2](figures/SSDTT_sample2.png?raw=true "SSD sample 2")
![SSD sample 3](figures/SSDTT_sample3.png?raw=true "SSD sample 3")

##### Udacity 40 epochs [[Weights](https://drive.google.com/open?id=0B6eUlGGeZ9wAWkNlV2VCeEJEaW8)]

Finally a last test with SSD on Udacity, where we obtain a really bad results, but a really high frame rate and the image samples, shows you what is happening. Seems that the dataset is bad balanced with the class truck and generates a lot of false positives in the images, dropping a lot the final scores.

Avg Precission = 0.662301095447
Avg Recall     = 0.295350784279
Avg F-score    = 0.408522453953
Average FPS: 111.89

![SSD plot](figures/SSDUdaciGraph.png?raw=true "SSD Experiment")

Image samples

![SSD sample 1](figures/SSDU_sample1.png?raw=true "SSD sample 1")
![SSD sample 2](figures/SSDU_sample2.png?raw=true "SSD sample 2")


<p align="right"><a href="#WSum">Back to summary</a></p>

<h1 id="Week 6">Week 6: Object Segmentation</h1>

## abstract

## Task summary
#### Task (a): Run the provided code and use the preconfigured experiment file (camvid_segmentation.py) to segment objects with the FCN8 model.
- Analyze the dataset.
- Evaluate on train, val and test sets.
#### Task (b): Read two papers
- Fully convolutional networks for semantic segmentation (Long et al. CVPR, 2015)
- Another paper of free choice.
#### Task (c): Train the network on a different dataset
- Set-up a new experiment file to image semantic segmentation on another dataset (Cityscapes, KITTI, Synthia, ...)
- Use the FCN8 model as before.
#### Task (d): Implement a new network
- Select one network from the state of the art (SegnetVGG, DeepLab, ResnetFCN, ...).
- Integrate the new model into the framework.
- Evaluate the new model on CamVid. Train from scratch and/or fine-tune.
#### Task (e): Boost the performance of your network
- meta-parameters tuning
- data augmentation

## Models

## Code changes and additions

## Hints 

## Datasets

### [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)

The CamVid dataset is a sequence inside an urban scenario composed by 368, 102 and 234 images for train, validation and test respectively. The image size is 480x360 and is composed by 11 classes:
- 0: 'sky',
- 1: 'building',
- 2: 'column_pole',
- 3: 'road',
- 4: 'sidewalk',
- 5: 'tree',
- 6: 'sign',
- 7: 'fence',
- 8: 'car',
- 9: 'pedestrian',
- 10: 'bicyclist',
- 11: 'void'
        
#### Analysing CamVid Dataset

We analyze the dataset as a part of the task a) to explain the composition of the train, validation and test images and the possible reason to the scores obtained in the experiments. We can see how the train images contain basically two sequences, one with a cloudy day that changes the illumination conditions in the image and another one with a sunny day, the train sequences aren’t correlative, have jumps and it’s more like a random image selection. The validation instead, it’s a pure sequence frame by frame with a sunny day. Finally the test it’s like the train, random image from sequences, one with a cloudy day and another sunny. In the experiments we saw a huge difference between train, validation and test. The validation is a bad sample to simulate the test, because is a sequence with the same illuminance conditions, even if the model scores well in the validation, when you try the test dataset the scores drops. So basically this dataset have a problem, needs more train data and a random sampling for the validation part.

### [Cityscapes](https://www.cityscapes-dataset.com/)

the Cityscapes dataset is a high resolution sequence of an urban environment composed by 2975, 500 and 1525 images for train, validation and test respectively. The image size is 2048x1024 and is composed by 20 classes:
- 0: 'road', #7
- 1: 'sidewalk', #8
- 2: 'building', #11
- 3: 'wall', #12
- 4: 'fence', #13
- 5: 'pole', #17
- 6: 'traffic light', #19
- 7: 'traffic sign', #20
- 8: 'vegetation', #21
- 9: 'terrain', #22
- 10: 'sky', #23
- 11: 'person', #24
- 12: 'rider', #25
- 13: 'car', #26
- 14: 'truck', #27
- 15: 'bus', #28
- 16: 'train', #31
- 17: 'motorcycle', #32
- 18: 'bicycle', #33
- 19: 'void' #34

### [Synthia rand cityscapes](http://synthia-dataset.net/)

This Dataset is provided from the virtual environment Synthia, consisting of a random sequences with a lot of elements in the image. The dataset consist in 7521, 471 and 941 images for train, validation and test respectively. The image size is 1280x760 and is composed by 20 classes:
- 0:  'road',
- 1:  'sidewalk',
- 2:  'building',
- 3:  'wall',
- 4:  'fence',
- 5:  'pole',
- 6:  'traffic light',
- 7:  'traffic sign',
- 8:  'vegetation',
- 9:  'terrain',
- 10: 'sky',
- 11: 'person',
- 12: 'rider',
- 13: 'car',
- 14: 'truck',
- 15: 'bus',
- 16: 'train',
- 17: 'motorcycle',
- 18: 'bicycle',
- 19: 'void'

## Metrics

accuracy - Evaluating pixel by pixel of the pixel class predicted versus the pixel clas from the ground truth.
jaccard - intersection over union between all the pixel from a class of the predicted model versus the ground truth.

## Experiments

In this section we analyze the different experiments performed in each dataset using our models implemented. All the experiments use adam optimizer with a learning rate of 0,0001.

### CamVid

We tested in this tiny dataset our differents models implemented Fcn8, segnet and ResnetFCN.With the ResnetFCN we tried some hyperparameters optimization to improve the results.

#### Fcn8

A first default test using the Fcn8, with pretrained weights, 100 epochs, the scheduler active in poly mode and no more options.

![Fcn8 plot](figures/fcn8Camvid.png?raw=true "Fcn8 Experiment")

This first attempt shows how the model learns fast and the accuracy grows in few epochs easily, the jaccard metric is difficult to increase due the high threshold because we want the overlap between the ground truth really similar. We score a 0.64 in the mean validation jaccard metric and the early stopping finished the experiment around the epoch 50.
The test results are the next:

acc: 0.879246986697
      loss: 0.463525073028
    0 (      sky      ): Jacc:  90.02
    1 (   building    ): Jacc:  78.43
    2 (  column_pole  ): Jacc:  13.89
    3 (     road      ): Jacc:  89.42
    4 (   sidewalk    ): Jacc:  67.27
    5 (     tree      ): Jacc:  73.02
    6 (     sign      ): Jacc:  30.88
    7 (     fence     ): Jacc:  30.56
    8 (      car      ): Jacc:  74.30
    9 (  pedestrian   ): Jacc:  36.01
       10 (   byciclist   ): Jacc:  37.86
       Jaccard mean: 0.565144366976

The final Jaccard mean in test is lower than the validation, this can be explained due the low number of train images, are insufficient to learn properly all the element to segment in the image. This problem is reflected with the classes with low jaccard score like poles, signs, fences, pedestrian and cyclists, because there are small elements in the image, more difficult to learn and where you need more data to learn properly them. Nevertheless, the extensive classes like road, sky, building, etc. They score high jaccard values because there are presents in all the images and are more easy to learn for the model.

#### Segnet

For the segnet model we don’t have pretrained weight, so we tried to perform some hyperparameter optimization to improve the final scores. We made a first step using 200 epochs, scheduler in linear mode.

![segnet plot](figures/segnetCamvid1.png?raw=true "segnet Experiment")

We can see how this model uses all the epochs to improve, may be because we don’t have pretrained weights it’s more difficult to learn if we compare with the previously fcn8 experiment. Also we can see the learning phase is quite irregular between epochs, more noticeable in validation with a lot of spikes, ending with a validation jaccard of 0.6, a bit less than fcn8.
With another step using data augmentation, only zooms and horizontal flips we can increase the score slightly:

![segnet plot](figures/segnetCamvid2.png?raw=true "segnet Experiment")

In 100 epochs more we increase the validation accuracy to near 0.64. If we compare directly with the results obtained in fcn8, for this model we needed 8 times more epochs to reach the same validation metrics, but have more margin to improve, because the training scores are crealy lower than the fcn8 mode, but without a pretrained model becomes a more difficult.
the test results are the next:
      acc: 0.846697468041
      loss: 0.481149245124
    0 (      sky      ): Jacc:  87.76
    1 (   building    ): Jacc:  71.66
    2 (  column_pole  ): Jacc:  15.65
    3 (     road      ): Jacc:  86.74
    4 (   sidewalk    ): Jacc:  61.62
    5 (     tree      ): Jacc:  65.76
    6 (     sign      ): Jacc:  19.10
    7 (     fence     ): Jacc:  18.52
    8 (      car      ): Jacc:  63.85
    9 (  pedestrian   ): Jacc:  30.65
       10 (   byciclist   ): Jacc:  31.96
       Jaccard mean: 0.502967009151

Clearly lower than the fcn8 in all classes.

#### ResnetFCN

The ResnetFCN uses more parameters because is wider than the other models, we need to be careful with the batch size reducing it between 2 and 4, depending of the input sizes.
A first test similar to the previous models, using pretrained weights, 100 epochs, scheduler in poly mode.

![ResnetFCN plot](figures/resnetFCNCamvid.png?raw=true "ResnetFCN Experiment")

We can compare the results directly with the Fcn8. This model learns fast, also adapts the train easily and have overfitting, they adapt a lot the jaccard measure and the gap between validation in huge. This have sense because is a resnet, resnet have redundancies and adapts the train really good. In general is better in all the aspects and the early stopping also finished around epoch 50. But what happen with test:
acc: 0.866687767545
      loss: 0.561303280754
    0 (      sky      ): Jacc:  89.92
    1 (   building    ): Jacc:  74.41
    2 (  column_pole  ): Jacc:  17.56
    3 (     road      ): Jacc:  91.13
    4 (   sidewalk    ): Jacc:  72.55
    5 (     tree      ): Jacc:  63.14
    6 (     sign      ): Jacc:  22.64
    7 (     fence     ): Jacc:   8.83
    8 (      car      ): Jacc:  70.45
    9 (  pedestrian   ): Jacc:  28.10
       10 (   byciclist   ): Jacc:  30.69
       Jaccard mean: 0.517666915173

We can see that in test is worst than Fcn8. Some classes have similar scores, road are better but someone like fences are worst. But theoretically, this model should be better. Basically the problem is the input preprocess, we don’t follow the one used for the pretrained model and this affect negatively. We tried a new approach using this new preprocessing for the mean and std that can be found inside the data_loader.py. Also we tried some other techniques, like use crops of the train instead of all the image, to improve the time performance and data augmentation with zoom and horizontal flips. We changed the scheduler to linear mode and run 300 epochs. The results are the next:

![ResnetFCN plot](figures/resnetFCNCamvid2.png?raw=true "ResnetFCN Experiment")

We can see how this time, using data augmentation and the proper preprocessing, the model learning and improve during all the epochs programed. The accuracy is adapted really fast and the jaccard scores improves along epochs. Thanks to the data augmentation, now the gap between validation and train is lower, so our model generalize better, reaching a 0.78 validation jaccard, a huge increment comparing with the other models. The test result are the next:
      acc: 0.908216732644
      loss: 0.372034771608
    0 (      sky      ): Jacc:  91.19
    1 (   building    ): Jacc:  81.88
    2 (  column_pole  ): Jacc:  32.25
    3 (     road      ): Jacc:  94.06
    4 (   sidewalk    ): Jacc:  79.34
    5 (     tree      ): Jacc:  74.04
    6 (     sign      ): Jacc:  38.98
    7 (     fence     ): Jacc:  25.55
    8 (      car      ): Jacc:  86.24
    9 (  pedestrian   ): Jacc:  56.63
       10 (   byciclist   ): Jacc:  60.85
       Jaccard mean: 0.655464245047

We obtain a huge increment in test also, but still really low compared to validation. Having problems with small elements.

A final experiment using this last one to try to improve more the scores, was made using all the input size instead of crops, without the preprocessing input said before and the same data augmentation. The results are interesting:

![ResnetFCN plot](figures/resnetFCNCamvid3.png?raw=true "ResnetFCN Experiment")

There is not a notorious improvement, but we can see how smooth our model becomes, where the variation in validation is really low between epochs, having a model really stable. But the model overfit the train more than improve the validation. The test results are the next:
acc: 0.914634088595
      loss: 0.352595471326
    0 (      sky      ): Jacc:  91.40
    1 (   building    ): Jacc:  82.65
    2 (  column_pole  ): Jacc:  34.50
    3 (     road      ): Jacc:  94.98
    4 (   sidewalk    ): Jacc:  83.30
    5 (     tree      ): Jacc:  75.38
    6 (     sign      ): Jacc:  42.61
    7 (     fence     ): Jacc:  32.84
    8 (      car      ): Jacc:  84.13
    9 (  pedestrian   ): Jacc:  59.29
       10 (   byciclist   ): Jacc:  64.70
       Jaccard mean: 0.677972789013

We can see in test an improvement, respect the previous experiment, making more meaningful the changes for this test.

##### Result Visualization

We have a function that print the results of the segmentation in each epoch, taking random samples from the validation and storing in disk. The image shows 4 images, at left we have the original image, next the ground truth overlapping the original image, next our detection and finally at right our detection overlapping the original image. Also at the bottom we have a legend with the colours associated to the classes.

- Epoch 1:
![Epoch 1](figures/resnetFCNEpoch0_sample1.png?raw=true "Epoch 1 ResnetFCN")
- Epoch 50:
![Epoch 50](figures/resnetFCNEpoch0_sample2.png?raw=true "Epoch 50 ResnetFCN")
- Epoch 100:
![Epoch 100](figures/resnetFCNEpoch0_sample3.png?raw=true "Epoch 100 ResnetFCN")
- Epoch 300:
![Epoch 300](figures/resnetFCNEpoch0_sample4.png?raw=true "Epoch 300 ResnetFCN")

We can see how the model at the beginning is a mess, but in the epoch 50 have a close result and the next epochs shows how this result is refined adjusting better to small objects.

### Cityscapes

We experiment with the Cityscapes dataset, using the Fcn8 and ResnetFCN. How they didn’t provide ground truth for the test dataset, we only have the score obtained for validation. But this dataset have more images for train and the validation in more appropriate for evaluate the model than the CamVid used.

#### Fcn8

We tried a simple test with Fcn8, like the one with CamVid, but rescaling the images that are really high to 512x256, using the same configuration, no data augmentation and weight pretrained the scores obtained are the next ones.

![Fcn8 plot](figures/fcn8Cityscapes.png?raw=true "Fcn8 Experiment")

We can see that the model learns, the accuracy grows easily for the train data and also the jaccard grows sufficient, but in validation we can see a huge gap, where the model is not generalizing well. Basically because we don’t use data augmentation and the data to train is reduced in a quarter the original size, losing a lot of information that can be relevant.

##### Visualization samples

![fcn8CityscpSamples1](figures/fcn8CityscpSamples1.png?raw=true "fcn8CityscpSamples1")
![fcn8CityscpSamples2](figures/fcn8CityscpSamples2.png?raw=true "fcn8CityscpSamples2")
![fcn8CityscpSamples3](figures/fcn8CityscpSamples3.png?raw=true "fcn8CityscpSamples3")

Some results obtained during the train of the Fcn8. first sample for the first epochs, next after 10 and the last one around the 40th epoch.


#### ResnetFCN

We tried two configurations in ResnetFCN, one like the first experiments without nothing special and maintaining the resize of 512x256 to compare with the Fcn8. The results are the next.

![ResnetFCN plot](figures/resnetFcnCityscapes.png?raw=true "ResnetFCN Experiment")

The main difference with the Fcn8 model is that this model overfit too much and from the epoch 20 and above starts to increase the validation error, clear sign of overfitting, because the train still learning and the jaccard reach high scores.

Another test is performed, this time using a resize to a half of the original image 1024x512 and we perform crops in the train of 512x512, we need to reduce the batch size to 2, if we want to fit in memory the train process. Using data augmentation and preprocessing the input like the best result obtained for CamVid, we obtain the next results in two consecutives runs.

![ResnetFCN plot](figures/resnetFCNCityescapes2.png?raw=true "ResnetFCN Experiment")

![ResnetFCN plot](figures/resnetFCNCityescapes2.png?raw=true "ResnetFCN Experiment")

The graphs shows how the model generalize better and the score for validation improves a lot, when we use data augmentation and a proper input processing with sizes and crops. The model have margin to learn and improve, but the epochs are really time consuming.

##### Visualization samples

![resnetFCNsample1](figures/resnetFCNsample1.png?raw=true "resnetFCNsample1")

![resnetFCNsample2](figures/resnetFCNsample2.png?raw=true "resnetFCNsample2")


<p align="right"><a href="#WSum">Back to summary</a></p>

## Reference
Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556. [Summary](Summaries/VGG.md)

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 770-778). 

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2818-2826). [Summary](Summaries/InceptionV3.md)
