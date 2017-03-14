# Scene Understanding for Autonomous Vehicles with deep neural networks
Master in Computer Vision - M5 Visual recognition

## Project Documentation
### Project slides
- [Google slides](https://docs.google.com/presentation/d/1AHFAFCaj7uQkiXKEfW8QbdqtRUyyvBKobA6NQJt4pjs/edit?usp=sharing) [In process]

### Project description
The goal of this project is to study the use of deep learning to semantically segment images and extract some knowledge of the scene. To do it we will use different types of neural networks.

- [Project Paper](https://www.overleaf.com/read/zbhrkkjvwkjv) [In process]

## Groups
 - [Group 6:](https://github.com/LLebronC/mcv-m5)
  - Jose Luis Gómez (joseluis-master@hotmail.com)
  - Luís Lebron(luis.lebron@e-campus.uab.cat)
  - Axel Barroso (axel.barroso@e-campus.uab.cat)
  - Hassan Ahmed
 
# Week 2: Object Recognition

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
  
## Reference
Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556. [Summary](Summaries/VGG.md)

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 770-778). 

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2818-2826). [Summary](Summaries/InceptionV3.md)
