# Scene Understanding for Autonomous Vehicles with deep neural networks
Master in Computer Vision - M5 Visual recognition

## Project description
The goal of this project is to study the use of deep learning to semantically segment images and extract some knowledge of the scene. To do it we will use different types of neural networks.

Download the [Overleaf document](https://www.overleaf.com/read/zbhrkkjvwkjv)

# Week 2: Object Recognition

## Abstract
This second week goal is prepare an object recognition classifier using the state of art of Neural Networks. As a first step for the M5 project, we need to identify properly the elements in a image, how the project is focused in Scene Understanding for Autonomous Driving, the elements to identify will be the relevants for this fuction, how they can be cars, pedestrians, traffic signs, etc. In order to train and validate our classfier we use Datasets provided with related traffic elements, this Datasets are TSingHua-TenCent 100K, INRIA pedestrians, Belgium traffic signs, Daimler pedestrians and KITTI. The classifier will be based on Convolutional Neural network, studying the actual state of art for image classification, training the best models of this last years and comparing the results.

## Framework composition

A framework in Python is provided to prepare easily the models, read automatically the provided Dataset, splitting in Train, Validation and Test, where with a configuration file we can manage all the parameters needed to train the CNN. The framework use Keras library and supports Theano and TensorFlow, making easy the compatibilily of both libraries and the creation or integration of models.
[TODO: EXPLAIN MORE ABOUT DE FRAMEWORK]

### Modules used

The following modules are being used for this week to understand properly the code and fix minor problems:
- Main file train.py: where we can see the calls of the different modules and the arguments to use when we run it.
- Configuration: module where a configuration file is read in order to automatize the preparation of the experiments.
- Models: model factory implementation to add easily new models to the framework
- Tools: where we can find the optimizer factory to use the diferent optimizers integrated to use in the CNN, also the dataset loader and related files.
- Callbacks: callback factory to personalize the callback to use when a experiment is running.

### Run an experiment

To run an experiment using TensorFlow we use the next command with optional parameters as input in []: CUDA_VISIBLE_DEVICES=1 python test.py -c config/TT100k_classfier.py -e ExperimentName [-s SharedPath (default:/data)] [-l localPath (defaut: /datatmp).
We can also define by default this input parameters in the train.py in the lines 81 to 88. This command read the Dataset from the shared path specified, make a local copy on the local path especified, read the configuration file and names the experiment.
Automatically the execution creates a Experiments folder on the local path and stores information relevant about this experiment, like wieghts of the model, results, log files, etc. The configuration file especified in the command is the main tool to configurate and parametrize the differetns experiments to do, selecting the CNN, Dataset, training configuration and data preparation.

### Integrate a new model

The steps to add a new model to the framework are:
- Define the model in Keras language and store it in /models.
- Go to the model_factory.py file import your model or main function that creates your model (ex: from models.resnet import ResnetBuilder), add the name of the model that you want to use in the configuration file to call it in the make function, add the name inside the if of the line 82.
- Define the call of your model, adding in the function make_one_net_model the entry needed, follow the other nets already implemented as example.

## Code changes and additions

- Added the densenet model using the implementation from: https://github.com/titu1994/DenseNet, now the denset net 40 is available. For using this model, please set densenet into the config file (model_name).
- Small change. Modified TensorFlow dowload link in the code/README.m file, from https://github.com/Theano/Theano to https://www.tensorflow.org/install/install_linux.
- Modified Optimizer_factory from Tools folder, added learning rate parameter to the Adam and SGD optimizers in order to be useful.
- Added new CNN models for classification: Resnet50 and InceptionV3. Using the Keras implementation are available in the configuration file in the parameter "model_name".
- Support for Weight decay integrated for this Models: VGG16, VGG19, Resnet50 and InceptionV3. The Weight Decay use L2 normalization and is activated when the value of the parameter "weight_decay" of the configuration file is higher than 0.
- Added the different resnet models using the implementation from: https://github.com/raghakot/keras-resnet, now the resnet 18, 34, 50, 101 and 152 are available, using resnetXX in the config file, the resnet50 from Keras is called using resnet50Keras.
- Added support for transfer learning from different datasets (different number of class) and different path for pretrain weights and test weights.

## Configuration Hints

- Densenet is really heavy, it goes out of memory very easily in our GPU. The minimum size is 32x32, 64x64 images work fine. 40 is the default depth of the net, this parameter can be changed into model_factory.py as an input parameter for the method build_densenet(..., depth = 'x').
- Resize dataset: remember that Resnet50 and inceptionV3 need a resize for the dataset in order to process the input image properly and work. Minimum input size [139,139] for InceptionV3 and [199,199] for Resnet50. When the dataset is smaller than this size like TT100K, use the reasize parameter close to the minimum in order to penalize minimally the computation time for epoch only in the models with this requeriments.
- TensorFlow try to allocate memory on the GPU even if don't fit firstly, if you use batch sizes inapropiate or resize the image to higher dimensions you can experiment a drastical increment in the epoch calculation time, so it's improtant to take into account.
- Use weight decay with lower values (exponent -4 or -5), high values penalyze the weight and allows less variations needing more time to train and higher loss values, still the CNN will learn, but not appropriately.

## Task summary
### Task (a): Run the provided code

### Task (b): Train the network on a different dataset

### Tasks (c) and (d): Implement a new network

### Task (e): Boost the performance of your network

## Tests Realized

### Weight Normalization
Tested the Weight normalization behaviour in the models trying a weight decay value of 0.1, 0.01 and 0.001 in VGG16.
- 0.1: High starting loss, around 150, each epoch reduce this loss constantly but the training step is really slow per epoch, reducing the loss in the validation data but poor changes in the accuracy score. Too slow and higher penalization for weight is not useful.
- 0.01: Similar to the previous case, but the lost start at 20 and drops slowly, being better but still too slow to be really useful.
- 0.001 and above: train fast, similar loss speed like without weight decay but with a better control of the train step and normalizing the weights.

### Train from Scratch VS ImageNet
In general, with sufficient time to execute, both method achieve similar accuracy if you choose proper optimizer configurations, the main difference is the starting point, from scratch the start loss are high and falls more slowly than use the ImageNet weight where the starting loss is lower. It's recommended to start with imageNet weight if the model support it to save execution time and can be useful to perform Fine Tune adding new layers of modifying the top of a model.

#### Fine Tune example

A Fine Tune example can be seen in the test Datasets section bellow, the experiment is with a ResNet 50 of Keras tested in TT100K Dataset. A first train step with the weight preload from ImageNet and frozen to keep the values during 19 epochs, where we can see how is slowly adapted due the learning of the top of the model part, the flatten layer and softmax. When is observed that the model can't adapt more the loss, we stop the training and reestart again unfrozen the layer from the base model part, allowing the learning. From this point (epoch 19) we can see how the loss drops until the saturation of the train.
If we compare with the ResNet 50 from scratch just bellow this one, there aren't notables differences at the end, ResNet learn quite good the Dataset with the proper learning rate and data augmentation, don't needing the ImageNet weights, but the fact that we used this weights improve sighlty the final score in validation due the differences in weight adaptation during the train phase, end with this little variation that benefits the Fine Tune case.

### Analysing different pre-processings

### Transfer learning to another dataset
We use transfer learning using VGG16 from the TT100K dataset to the BelgiumTSC dataset. 

- train VGG16 in the BelgiumTSC from scratch 

![VGG16 Scratch plot](figures/plotVGG16_BTS_scratch.png?raw=true "VGG16 BelgiumTSC Scratch Experiment")

- train VGG16 in the BelgiumTSC with pretrain weights 

![VGG16 Pretrain plot](figures/plotVGG16_BTS_pretrain.png?raw=true "VGG16 BelgiumTSC Pretrain Experiment")

### Analysing data augmentation

### Practical differences between Adam and RMSProp

### Techniques to improve the network results

### Dataset Experiments

#### TSingHua-TenCent 100K (TT100K)

This Dataset can be find here: http://cg.cs.tsinghua.edu.cn/traffic-sign/tutorial.html It's a Dataset of traffic signs of 64x64, that consist in 16527 images for train, 1655 for validation and 8190 for test, being the most challenging one for us with 220 classes of signals to identify and for this reaon is the preferent Dataset to test our Models.

Ranking best result of the models on TT100K daset: https://drive.google.com/open?id=1DPU7IIxMk3xZAEAY_5cH4X0YRmexPwH_1sK1yAKgDQs
Inside the document we can find the weight and configuration files associated to each experimented and can be recreated in this code.

![CNN Ranking](figures/Ranking.png?raw=true "CNN Ranking Experiment")

If we analyze the results obtained, in general we can see that the test Dataset for the TT100K should have a really similar properties like the train, because for all models the accuracy obtained is really high. In other hand, we can pay attention to the loss rating where the differences are noticed and also the validation accuracy. For this case we can see how the ResNet 50 Keras is the best model, the lowest loss and highest accuracy for validation. We should say that this experiments aren't homogenious, someone has more dataaugmentation, others ones don't have mean and std substraction, ImageNet weight preload, etc. All this factors makes little differences, but in general this are the top values obtained for each experiment and model.

##### VGG16 Keras
![VGG16 plot](figures/plotVGG16.png?raw=true "VGG16 Keras Experiment")

The main configuration for this test was using preload weight for ImageNet, optimizer Adam with 1-E05 of learning rate, weight decay of 1-E04, rescaling, std and mean substraction without Data augmentation. The graphic shows the evolution of the model until the early stops callback end the training step. The models starts with a low loss due the ImageNet weights.

##### VGG19 Keras
![VGG19 plot](figures/plotVGG19.png?raw=true "VGG19 Keras Experiment")

For this experiment the configuration are preload weight from ImageNet, only rescale to the input data and a data augmentation with 10 dregrees of rotation and shifts, shear and zoom of 0.1. We can see that the lowest loss for validation was between epochs 40-50, but the early stopping was defined using the validation accuracy in this case, scoring the best near the epoch 91. This example is similar to the previous one with VGG16, score a little more validation due the generalization of the data augmentation.

A common propiety of the VGG models is the amount of parameters that they have, needing a lot of memory and having heavy weight files. In general they allow small input images and are fast due the few layers in comparison to the other models. But the validation loss shows that this model is less stable and generalize a little worst than Resnet for example.

##### ResNet 50 Keras

For this example we should two experiments involving the ResNet 50 from the Keras implementation, one with Fine Tune and other one from scratch. In general this ResNet model is really smooth learning, don't have a high variantion within epochs in validation like VGG, even when the loss is a little higher, the scores in accuracy are close. The main problem is that they need a minimum input size of 197x197, needing to resize the Dataset input and increasing the computation time for epoch.

- with Fine Tune:

![ResNet 50 Keras plot](figures/plotResnet50Keras.png?raw=true "ResNet 50 Keras Experiment")

- From Scratch:

![ResNet 50 Keras plot](figures/plotResNet50.png?raw=true "ResNet 50 Keras Experiment")

##### ResNet 152
![ResNet 152 plot](figures/plotResnet152.png?raw=true "ResNet 152 Experiment")

##### InceptionV3 Keras
![InceptionV3 plot](figures/plotVGG19.png?raw=true "InceptionV3 Keras Experiment")

##### DenseNet 40 
![DenseNet plot](figures/plotDenseNet.png?raw=true "DenseNet 40 Experiment")

#### INRIA pedestrians

##### InceptionV3 Keras
![InceptionV3 plot](figures/plotInceptionINRIA.png?raw=true "InceptionV3 Keras Experiment")

##### VGG19 Keras
![VGG19 plot](figures/plotVGG19INRIA.png?raw=true "VGG19 Keras Experiment")

#### Belgium traffic signs (TBSC)

##### ResNet 50 Keras
![ResNet 50 Keras plot](figures/plotResnet50BTSC.png?raw=true "ResNet 50 Keras Experiment")

#### KITTI

##### ResNet 50 Keras
![ResNet 50 Keras plot](figures/plotResnet50KITTI.png?raw=true "ResNet 50 Keras Experiment")

## Project slides
- Google slides for [Week 1](https://docs.google.com/presentation/d/1jiS8scHFZGNVeYUV8wJpG2AjLw9J-iSCpzYKUXmoAWQ/edit?usp=sharing)
- Google slides for [Week 2](https://docs.google.com/presentation/d/1AHFAFCaj7uQkiXKEfW8QbdqtRUyyvBKobA6NQJt4pjs/edit?usp=sharing)
- Google slides for Week 3 (T.B.A.)
- Google slides for Week 4 (T.B.A.)
- Google slides for Week 5 (T.B.A.)
- Google slides for Week 6 (T.B.A.)
- Google slides for Week 7 (T.B.A.)

## Groups
 - [Group 6:](https://github.com/LLebronC/mcv-m5)
  - Jose Luis Gómez (joseluis-master@hotmail.com)
  - Luís Lebron(luis.lebron@e-campus.uab.cat)
  - Axel Barroso (axel.barroso@e-campus.uab.cat)
  - Hassan Ahmed
  
## Reference
Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556. [Summary](Summaries/VGG.md)

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 770-778). 

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2818-2826). [Summary](Summaries/InceptionV3.md)
