# Phase 4 Project

Image Classification with Deep Learning

## Introduction

For my Flatiron School Phase 4 Project I will be classifying x-ray images of pediatric patients to determine whether they have pneumonia or not. Pneumonia is an inflammatory condition in the lungs that affects small air sacs known as alveoli. Pneumonia can be caused by viruses, bacteria, and occasionaly other microorganisms such as fungi and parasites. Symptoms typically include cough, chest pain, fever, and difficulty breathing. Pneumonia affects everyone differently, but children, the elderly, and those with pre existing health conditions are the most at risk. Diagnosing pneumonia using traditional physical exams can be difficult so chest x-rays are commonly used, but interpreting chest x-rays can be difficult as well due to the the locality and severity of the case determining whether it is distinguishable by the naked eye or not. Pneumonia affects roughly 450 million people per year globally and results in around 4 million deaths per year. Antibiotics and vaccines have greatly improved survival rates, but pneumonia is still considered a leading cause of death in many developing countries. The economic costs associated with pneumonia are debated with estimates ranging from $27 billion per year globally, up to almost $20 billion per year just in the United States. The average hospital charges for treating pneumonia in the US are around $25,000 and as high as $125,000, and the average emergency room visit for pneumonia costs $950. A robust deep learning model that can classify x-rays with high accuracy has many potentially beneficial applications. To medical personal in assisting to diagnose pneumonia cases, it could reduce their time spent analyzing radiographic images. To patients, as an earlier diagnosis may lead to an improved outcome and less time sick. And also to hospital administrators and insurance companies, as detecting and treating pneumonia cases earlier and more efficiently could save lives, time, and resources. 

The data used for this project was collected and made available by Kermany et all via [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/3) and was downloaded from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## Technologies
* The data cleaning, organizing, and modeling for this project can be found in the jupyter notebook titled "Phase4Project", it is located in this page's repo and was created using Python 3.7.6
* There is a keynote presentation for non-technical audiences available under the file name "Phase4Presentation.pdf"

### Necessary libraries to reproduce and run this project are:

* OS
* CV2
* Pandas
* NumPy
* MatPlotLib
* Seaborn
* SKLearn
* Keras
* Tensorflow

## Objectives

* Explore and analyze dataset using visualizations to identify image and label distributions as well as image examples.
* Preprocess images using normalization and augmentation techniques.
* Model the image data using convolutional neural networks. Iterate through a baseline model, a regularized model, and a model built using transfer learning.
* Validate model performance by making label predictions and evaluating predictions against the test data. Explain the strengths and weaknesses of the model's predictive power. 

## Methodology

A neural network is a computing system inspired by biological neural functions in animal brains. They are composed of a system of nodes (neurons) that send signals from a given input in the form of linear functions back and forth to generate an output represented as a function of the sum of the inputs. The architecture of a neural network is defined by the the number of neurons in a layer, and the number of layers in the network. A neural network comprised of multiple layers is known as a multi layer perceptron (MLP). The amount of layers per network and neurons per layer is problem specific and the there is no definitive standard for their configuration, as each problem requires experience and experimentation from the practitioner engaged in modeling a problem. Once a network is provided information it uses a user specified activation function to connect the neurons (data features) and calculate their weights and biases. Once the weights are calculated with the goal of generating a probability for each data observation and its label, a cost function using gradient descent attemps to minimize an error rate by optimizing the weights and biases through a recursive series of forward and backward propogation. During propagation gradient derivatives are used to calculate the difference between the calculated and desired outcome (known as error) and this is repeated until a local or global minima is reached. The summary of all errors is known as loss and the objective of the model is to minimize loss while maximizing a specific evaluation metric, which is accuracy for this problem of image classification. 

Convolutional neural networks (CNN's) are deep neural networks that specialize in processing image data by assigning weights and biases to specific features in the image that differentiate it from other images. They are initialized with a densely connected layer and then multiple convolved and pooling layers are stacked and a final dense layer is added at the end to learn the decision function based on the convolved and transformed informative inputs. 

Images are represented by three dimensional tensors with each matrix of the tensor corresponding to a color (Red, Green, Blue) and each cell in the matrix containing a pixel value. Processing the amount of information contained in an image, especially a high resolution image, would generate an exponential explosion of parameters and weights that would be far too computationally intensive for any conventional computer to process. The solution to this problem is the implementation of convolutional and pooling layers. Convolutional layers pass a smaller matrix filter over the image tensor and use matrix multiplication to generate a new lower dimension feature output. The lower dimension feature output is then passed to a pooling layer which reduces the spatial size of the convolved feature. The convolutional layer finds the borders and shapes of the image and then the pooling layer takes the summary of the convolutions and down samples them into lower dimensional representations that are computationally viable. Together the convolutional and pooling layers are able to extract low and high level features from the image while at the same time reducing dimensionality. Before being passed to the densely connected output layer the images are flattened into a column vector, and the flattened output is finally fed into a feed-forward neural network.  

The convolutional neural network will take a training collection (images pre-labeled as Normal or Pneumonia) as input, transform their information into a proccesable format, and then form a probability weighted association between the image information and a label. Once the model is configured, compiled, and fit, it will be fed images with no attached label and will use the features learned in training to generate a predicted label. The predictions will then be evaluated and the number of correctly predicted labels divided by the total number of images in the testing dataset will be the models accuracy. 

## Table of Contents

* [Exploratory Data Analysis](#EDA)
* [Preprocessing](#Process)
* [Modeling](#Models)
* [Model Evaluation](#Eval)
* [Conclusions](#Conclude)
* [Resources](#Resource)

<a name="EDA"></a> 
## Exploratory Data Analysis

I used simple bar plots to inspect the distribution of labels among the data, and a simple plotting method to inspect example images from the dataset. The provided dataset is comprised of 5,216 training images, 16 validation images, and 624 testing images. The training images are divided into around 25% normal x-rays and close to 75% containing pneumonia, and the testing images are divided into about 35% normal x-rays and 65% containing pneumonia. 

![distribution](https://raw.githubusercontent.com/joshblumer/dsc-phase-4-project/main/images/class_distribution.png)

Having an asymmetrical distribution of label examples is known as class imbalance and often leads to a model that over-fits to the over represented class. The partitioning of train/validate/test sample sizes is a problematic distribution as well, as an ideal dataset is composed of approximately 70-80% training data, 10-15% validation, and 10-15% testing. I attempted modeling the data in the provided distribution to gauge baseline model performance, and then discarded the provided validation set and generated a proportional validation set using the training data afterwards. 

<a name="Process"></a>
## Preprocessing

The images provided in the downloadable dataset were color images which generally have much higher pixel density and values that can be computationally expensive and slow to process if not using an advanced GPU or powerful cloud server. Neural networks process normalized data much faster and converge to a stable solution more frequently using normalized data as well so I manually grayscaled and normalized the images by rescaling them into a 256 pixel format. 

![xray](https://raw.githubusercontent.com/joshblumer/dsc-phase-4-project/main/images/xray_example.png)
* As you can see in the x-rays, detecting pneumonia with the naked eye can be challenging, especially to the layman without medical training and education. 


Many popular and accurate CNN's were built being trained on tens to hundreds of thousands of images so the roughly six thousand images provided in this dataset is considered sparse. Neural networks are a very data intensive modeling method and the more data and more variance in the data the stronger your model will be and less likely it is that the model will over-fit to the training examples. The Keras library provides a tool for image processing called 'ImageDataGenerator' that allows you to augment your images by rotating, zooming, shifting, shearing, and flipping them, which can provided several variants of the same image in your data providing synthetic variance that helps your model distinguish differences among images. After modeling the dataset in its provided format I augmented the images using 'ImageDataGenerator' and re-fit them again with the augmented images.

<a name="Models"></a>
## Modeling 

Working on a data science project is an iterative process that involves recursively making changes to your preprocessing and feature selection techniques, as well as your modeling techniques, in an effort to optimize the model to give you the best performance. Neural networks can be more difficult to fine tune than other modeling techniques due to the unsupervised nature of the hidden layers and inability to use grid searches to find optimal hyper-parameter values. Neural network architecture contains many tuneable hyper-parameters such as number of layers and neurons, activation functions, loss functions, algorithm optimizers, learning rates,  evaluation metrics, and regularization techniques and values. Given the many possible combinations of hyper-parameters I decided to initialize one configuration of layers and neurons and hold that configuration as well as the activation and loss functions, optimizer, learning rate, and evaluation metric constant in an attempt to more accurately assess the data augmentation, distributions, pixel size, and regularization techniques as explanatory variables for model performance. The caveat to the listed configuration was an attempt at transfer learning at the end of the modeling phase. These decisions were made due the time and hardware constraints I was working under throughout the duration of the project as the fitting portion during modeling is very time consuming, and because my personal priority when approaching this project was to learn about and understand as much of the modeling process and hyper-parameter relationships as possible as oposed to producing as optimal of scores as possible.

My model infrastruce consists of a CNN with 4 hidden layers, each with different values of neurons, the rectified linear unit (ReLu) activation function for all layers except the output layer which is a sigmoid function which outputs a value between 0-1 and is necessary for a binary classification. I used the adaptive moment estimation (AdaM) convergence algorithm due to its popularity and being suggested in many research articles. I implemented a fixed 0.001 learning rate in order to help isolate the explanatory variables I listed above, and I used binary cross-entropy and the loss function and accuracy as the evaluation metric due to suggestions from educational literature. 

### Base Model
I began modeling using the provided dataset in the format it was downloaded with only 16 validation examples. I first modeled it with no augmentation or regularization and it returned an accuracy of 74.03% and a loss of 200.31%. I then added augmentation with no regularization and that model improved drastically to 89.90% accuracy and 26.77% loss. Next I added a 20% dropout regularization to 3 layers which yielded 90.22% accuracy and 27% loss. I began modeling with a 150x150 image size and then modeled 256x256 to compare outcomes. The larger pixel size returned a reduced score of 87.50% accuracy and 31.42% loss. 

![base](https://raw.githubusercontent.com/joshblumer/dsc-phase-4-project/main/images/base_model.png)
* The first model with no augmentation or regularization displays training and validation loss and accuracy scores that are both separated by wide margins indicating the model is very over-fit. The training curves both follow an exponential pattern which indicates a good learning rate.

### Validation Split Model
The next iteration of modeling involved disgarding the included validation data and generating a more proportional validation set using the Keras 'ImageDataGenerator' 'validation_split' argument. I generated a 20% validation set out of the training set which gave me 4,173 training examples, 1,043 validation, and the testing set of 624 images was not changed. Due to the improved performance of adding dropout regularization in the first modeling iteration I began with that same configuration which returned 91.50% accuracy and 30.71% loss. A slight improvement in accuracy and an increase in loss, so not a significant performance increase overall. I then modeled the same configuration again using the 256X256 pixel size and performance decreased to 81.41% accuracy and 38.47% loss. 

![split](https://raw.githubusercontent.com/joshblumer/dsc-phase-4-project/main/images/split_model.png)
* The split model training and validation curves follow each other more closely but still indicate a less than desirable fit. The training curves are not as consistent and show volatile jumps between epochs which means the learning rate may be calculating the weights too quickly.

### Heavily Regularized Model
The last iteration using the same model configuration I'd used with the previous models included an added L1 and L2 (weight decay) regularization parameter in the last convolved layer. L1 and L2 regularization add a penalty parameter to the cost function that keeps the weights from growing too large and forces the network to be more simple. The goal of adding regularization is to reduce over-fitting of the training data and increase how much the model is able to generalize to the testing data. Adding an L1 parameter returned an accuracy of % and a loss of %, and adding an L2 parameter returned an accuracy of % and a loss of %.

![regular](https://raw.githubusercontent.com/joshblumer/dsc-phase-4-project/main/images/regularized_model.png)
* The regularized models loss indicates a better fit but the accuracy hit a low ceiling and isn't fit as well. The loss lambda hyper-parameter needs to iterate through more values to attempt to find a better fit.   

### Transfer Learning Model
My final modeling iteration was an attempt at transfer learning. Transfer learning is a machine learning method that uses the architecture and parameters of a previously trained model on a new dataset. Pretrained models can be very effective with smaller datasets like the one I was working with, especially in the context of predictive modeling problems that use image data as input. I used the InceptionV3 pretrained model from Keras and added a new densely connected layer on top of the output. There are two different methods you can implement with pre-trained models. The first is to use the pre-trained model as a feature extractor; in this instance you freeze the pre-trained models convolutional blocks so that their weights aren't updated after each training epoch. The second is to use the pre-trained model as a fine tuner, in this instance you unfreeze however many blocks you'd like and once unfrozen those layers weights will be updated with backpropagation in each training epoch, this usually requires reducing the learning rate to avoid getting stuck at a local minima, and can increase training time by a large factor. I implemented the first instance of transfer learning and froze all trainable blocks and then added one layer initially, which was followed by adding two additional densely connected layers with 20% dropout. I first added an L1 penalty parameter which was then followed by adding an L2 penalty parameter. The first model with only dropout regularization returned an accuracy of 88.94% and a loss of 29.75%. After adding L1 and L2 penalties the accuracy dropped to 86.38% and the loss increased to 40.30%. 

![transfer](https://raw.githubusercontent.com/joshblumer/dsc-phase-4-project/main/images/transfer_model.png)
* The loss follows a near perfect exponential curve and is closely fit but the accuracy still displays a poor fit as well as inconsistent changes between epochs. A slower learning rate may help, but his can also be attributed to the class imbalance and the model needing more regularization. 

<a name="Eval"></a>
## Model Evaluation 

The best performing model I was able to construct was the synthetic generated validation set iteration with data augmentation and dropout regularization applied. This model returned 92% accuracy and 31% loss. To get a better idea of the models strengths and weaknesses I generated predictions and then evaluated those predictions using a confusion matrix and classification report. 

![report](https://raw.githubusercontent.com/joshblumer/dsc-phase-4-project/main/images/class_report.png)
* Precision was the model's weakness as it scored 81.20%. It predicted 190 cases of pneumonia accurately out of 234 total pneumonia predictions giving us 44 false positives, meaning those 44 images were predicted to have pneumonia but did not. The model's strength was recall, which was 95.48%. The model correctly identified 190 out of 199 cases of pneumonia, which left us with 9 false negatives, cases of pneumonia that were classified as normal. In the context of solving a classification for this particular dataset, we would much prefer a false positive to a false negative because the consequences of having pneumonia and the model predicting the image as normal are much more severe than predicting pneumonia that isn't there. Further attemps to improve this model should prioritize optimizing the recall to reduce the number of pneumonia cases classified as normal. 

<a name="Conclude"></a>
## Conclusions

This project was a great introduction to neural networks and image classification, and given the time and processing constraints I was working under I am very pleased with the outcome. As mentioned previously there are many hyper parameters to tune in the configuration, augmentation, and regularization of neural networks which gives you an exponential number of possible outcomes. After initially iterating through multiple hyper parameter values as mostly random guesses (reminder that you can't implement an exhaustive grid search with neural networks) and notating the unpredictable and complicated interactions certain hyper-parameters had, I realized I needed to make educated guesses and began consulting published research papers and articles covering CNN topics, which refined my modeling approach and had a positive impact on the outcome. One element about working on this project that surprised me was the very large difference in modeling performance that augmenting the images made. If I had more time with this project I would have liked to model a range of learning rates to compare their impact on model runtimes and performance, and I would have liked to attempt more transfer learning models as feature extractors and fine tuners. Trial and error is an important facet of any machine learning problem and that is especially true when working with neural networks, I look forward to learning more about them in the future and how to optimize them for the best performance.

<a name="Resource"></a>
## Resources

https://www.wikidoc.org/index.php/Pneumonia_chest_x_ray 
https://en.wikipedia.org/wiki/Pneumonia 
https://en.wikipedia.org/wiki/Artificial_neural_network 
https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/ 
https://gist.github.com/RyanAkilos/3808c17f79e77c4117de35aa68447045
https://datascience.stackexchange.com/questions/29719/how-to-set-batch-size-steps-per-epoch-and-validation-steps
https://towardsdatascience.com/image-data-generators-in-keras-7c5fc6928400
https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/ 
