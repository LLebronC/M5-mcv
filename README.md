# Scene Understanding for Autonomous Vehicles with deep neural networks
Master in Computer Vision - M5 Visual recognition

## Project description
The goal of this project is to study the use of deep learning to semantically segment images and extract some knowledge of the scene. To do it we will use different types of neural networks.

Download the [Overleaf document](https://www.overleaf.com/read/zbhrkkjvwkjv)

# Week 2: Object Recognition

## Asbtract
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

## Configuration Hints

- Densenet is really heavy, it goes out of memory very easily in our GPU. The minimum size is 32x32, 64x64 images work fine. 40 is the default depth of the net, this parameter can be changed into model_factory.py as an input parameter for the method build_densenet(..., depth = 'x').
- Resize dataset: remember that Resnet50 and inceptionV3 need a resize for the dataset in order to process the input image properly and work. Minimum input size [139,139] for InceptionV3 and [199,199] for Resnet50. When the dataset is smaller than this size like TT100K, use the reasize parameter close to the minimum in order to penalize minimally the computation time for epoch only in the models with this requeriments.
- TensorFlow try to allocate memory on the GPU even if don't fit firstly, if you use batch sizes inapropiate or resize the image to higher dimensions you can experiment a drastical increment in the epoch calculation time, so it's improtant to take into account.
- Use weight decay with lower values (exponent -4 or -5), high values penalyze the weight and allows less variations needing more time to train and higher loss values, still the CNN will learn, but not appropriately.

## Task summaty
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

### Datasets test
#### TSingHua-TenCent 100K (TT100K)

Ranking best result of the models on TT100K daset: https://drive.google.com/open?id=1DPU7IIxMk3xZAEAY_5cH4X0YRmexPwH_1sK1yAKgDQs

##### VGG16 Keras
![VGG16 plot](figures/plotVGG16.png?raw=true "VGG16 Keras Experiment")

##### VGG19 Keras
![VGG19 plot](figures/plotVGG19.png?raw=true "VGG19 Keras Experiment")

##### ResNet 50 Keras
![ResNet 50 Keras plot](figures/plotResnet50Keras.png?raw=true "ResNet 50 Keras Experiment")

##### ResNet 152
![ResNet 152 plot](figures/plotResnet152.png?raw=true "ResNet 152 Experiment")

##### InceptionV3 Keras
![InceptionV3 plot](figures/plotVGG19.png?raw=true "InceptionV3 Keras Experiment")

##### DenseNet 40 
![DenseNet plot](figures/plotDenseNet.png?raw=true "DenseNet 40 Experiment")

#### INRIA pedestrians

#### Daimler

#### Belgium traffic signs (TBSC)

#### KITTI

## Project slides
- Google slides for [Week 1](https://docs.google.com/presentation/d/1jiS8scHFZGNVeYUV8wJpG2AjLw9J-iSCpzYKUXmoAWQ/edit?usp=sharing)
- Google slides for [Week 2](https://docs.google.com/presentation/d/10zmaSSAL-29dpJZ-WZNhcl8yIUSg30iNnNcKlxDesTc/edit?usp=sharing)
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
