# Machine-learning-models
PRETRAINEND MODELS

https://github.com/aisha-colab/Machine-learning-models/tree/main
Use google colab for this work
Architecture of CNN 
The working of CNN on the images taken by two probes (HVPS and 2D-S)
4.1. Architecture of CNN: In image classification each pixel plays an important role in make correct prediction in normal feed forward system the conversion of two-dimensional image into single array and feeding it makes the image loss its importance so to overcome this drawback convolutional neural network comes into play as it provides convolutional layer along with pooling layer.


Convolutional layer: As the name suggest it operation called convolution in which there is feature extraction done by applying this operation at specific kernel and specific pattern. Specific patterns of the images such as edges, dimensions are detected by filters.
There are multiple filters in convolution layer and per filter gives on feature map as output. Relu activation ids applied after convolution in order put zero for all non-negative values and to introduce nonlinearity.

 
 
Pooling layer: After convolutional layer this layer is used to extract most dominant features of the image either by doing Max pooling or avg pooling depending on the required of the image dataset.

Fully connected layer: After the above two layers there Is fully connected layer which is like feed forward neural network in which the output of both the above layers are flatten into 1D array.
PRETRAINED MODELS 
VGG16
VGG19
RestNet50
Inception-V3
EfficientNet
XCeption



1Loading and pre-processing the data:

As there are various pretrained models involved in this paper the pre-processing of data was very important work for getting high accuracy, data argumentation which is different for both the dataset the most appropriate argumentation is used after trying out various augmentation such as width_shift_range,shear_range_,vertical flip, height _shift _range etc the best worked with HVPS dataset was just zoom_range of 0.4 and shuffle=true in training data whereas no argumentation used in test and validation and for 2D-S zoom_range of 0.2,rescaling,shear_range=0.2,horizontal_flip was used in training dataset and rescaling in test set is also done. The amount of data required to train the model is reduced by fine tuning and transfer learning but then also there might me nit enough data to. Perform training on it and the data set may be imbalanced so to the stated above problems are solve by data augmentation. The work done by augmentation is it to maintain the features of original training set and resulting in size increase of training data.

2 Defining the model and its architecture:

 After the pre-processing the data the is defining the model, as this paper comprises of various pretrained models, so the compiling of each model is done and to get the high accuracy which layers to freeze in the model are to consider very minutely along with the correct optimizer in compiling. After using optimizer such as RMSprop and Gradient Descent the accuracy at time of training was very low around for some of the pretrained models whereas when experimented with ‘adam’ the accuracy was very much increase as shown in the result section. Adam has the best properties of AdaGrad and RMSProp algorithms inculcated in it which can handle limited drop on turbulent problem.


3 Training the models: 

For training the model the training images and Validation images with their corresponding true labels are taken then according to the there are no of epoch used in this paper using 25 epochs for all the models and on both the dataset. The accuracy of training accuracy, validation accuracy, training loss and validation loss is seen on running the trained model and the plots shown in the results justify that training the model on this value of epoch gives clear understanding of the accuracy and loss results.

4 Estimating the model’s performance: 

The performance of on test data is seen in the two datasets by the confusion matrix where each correct classes prediction values are shown in the diagonal elements and precision, recall and f1 score given in the classification report makes the understanding of the prediction more appropriate. The classification reports and confusion matrix are discussed in the result section.
The dataset which is used in this paper is having images taken from two probes HVPS and 2D-S. For using the pretrained models we have import libraries same like libraries imported for VGG16 as shown below:


5 Example (VGG-16)
#from keras. applications.vgg16 import VGG16


 
HVPS having total of 1409 images. Training/validation/test splitting results as follows

 
2D-S having total of 4190 images. Training/validation/test splitting results as follow

There is different image argumentation done for both the dataset for there is argumentation in training data and no argumentation done on validation data. The below way shows the pre-processing of HVPS and 2D-S dataset.
Resizing the images to (224,224) as this shape was giving good results, batch size of 32 and class mode is used for both the dataset is categorical as there is 5 classes for HVPS and 7 classes for 2D-S. 


After importing the below libraries for of various pretrained models same as it is shown for VGG16:
 

Then pre-processing the data (train, valid and test) for HVPS AND 2D-S as below where it includes zoom_range and shuffling in HVPS 

For the train data by using the ImageDataGenerator following are the augmentation used
Zoom range=0.4 

Path of train validation and test are used for getting the data from that location(HVPS and 2 D-S)
‘/content/HVPS /train’
‘/content/HVPS /val’
‘/content/HVPS /test’


 along with the target shape of 224x224 batch size of 32 class mode is categorical as 5 
 There is shuffling done only to the train data

2D-S rescaling, shuffling, zoom_range, shear_range, horizontal_flip.

For the train data by using the ImageDataGenerator following are the augmentation used
Rescale=1./255
Shear_range=0.2
Zoom_range=0.2 and horizontal flip whereas for test data only rescaling of1./255 

Path of train validation and test are used for getting the data from that location(HVPS and 2 D-S)
‘/content/2D-S/train’
‘/content/2D-S /val’
‘/content/2D-S /test’


 along with the target shape of 224x224 batch size of 32 class mode is categorical 7 classes are to be classified.

 There is shuffling done only to the train data



After pre-processing the data set provided index to the classes of each data set.

HVPS  

2D-S
 

VGG16 is trained model with lots of classes so not retraining the pre-existing weights by putting the top_layer as false then freezing the training layer did not add any new fully connected layer in for the both the dataset just added 5 dense layers for HVPS and 7 for 2D-S with ‘SOFTMAX’ activation as both the dataset are multiclass classification, then merging the pretrained model layer with the existing layer.
Then compiling the model with the correct optimizer(adam) for multiclass classification the loss is categorical _cross-entropy. Then finally trained the model on 25 epochs and saving the model for all the pr-trained models used that VGG16, VGG19, RestNet50, Inception-V3, Xception, EfficientNet.
Accuracy test on the test dataset for various models in the form of confusion matrix and classification report which consist of recall, precision, f1 score and support. The diagonal elements in the confusion matrix shows how many true labels are correctly predicted.

After defining the train, valid and test for pretrained models the fine tuning done for VGG-16,VGG-19,RestNet50,Inception-V3,EfficientNet,XCeption where for both the datasets following are the steps done in tuning them:

1)loading the required libraries such as from tensorflow.keras.applications.vgg19 import VGG19 similarlily for all the other models.
2)using ImageNet weights and not trained the pre-existing weights.

3)Flatten the layer(no extra layers are used in models VGG-16,VGG-19,RestNet50 and EfficientNet whereas in Inception -V3 Add a fully connected layer with 1,024 hidden units and ReLU activation and XCeption added layers are GlobalAveragegpooling and dense layer of 256 neuron and ReLU activation, for HVPS and drop layer is in 2D-S used.

4)Last dense layer with numer of classes in each dataset (HVPS 5 and 2D-S 7) with activation ‘SoftMax’.

5)Compiling the model using adam compiler and loss as categorial_crossentropy.

6)finally fitting the models using epoch =25 and step per epoch 20.

7)testing the accuracy and the confusion matrix with plots and classification reports for all models.

6 Image augmentation 
 The quantity of data needed to train a model is significantly decreased by using transfer learning and fine-tuning. There may not be enough data in some situations, though, to simply fine-tune deep learning models. Furthermore, even if there are sufficient data, the dataset may still be too unbalanced to allow for optimal model performance. 
Image augmentation, a technique for artificially boosting the amount of data by creating additional data points from existing data, is effective for overcoming the aforementioned issues. 
By randomly altering the original dataset, image augmentation produces a dataset that is identical to the original training dataset but fragmented. The constraint on the changing dataset is that the image must retain the same ground truth label following the augmentation.
a)A rotation range spins the image at random between 0 and 360 degrees in the clockwise direction. 
b) The width shift range  is the width by which we can move the image left or right and range from 0.0 to 1. 
c) The height shift range randomly moves the pixels up or down in the vertical plane. 
d) The shear range causes an axis-based distortion of the image, which is typically used to create or correct perception angles. 
e)Zoom range adds additional pixels to the image and zooms it at random. The input (left side) image in the example above receives a random zoom enhancement with a range of 0.5 to 1.0%. and
The augmentation method used depends on the task. Changing colour is not a useful method in our case, thus rotating or flipping is more beneficial
7 Evaluation Metrics:

For evaluating the performance of each pretrained model used in the dataset there is use of confusion matrix which gives the prediction of test class of the dataset.

1 Confusion Matrix:

It’s the matrix which compares the true value from the predicted value in the form of matrix here there are 5 and 7 classes for the HVPS and 2 D-S so the matrix formed are of order 5x5 and 7x7,so with the help of confusion matrix we can understand how the models are behaving on the dataset then interpreting it results. It can be with or without normalizations meaning that prediction value of the class is made in case of without normalization whereas there is value for prediction in form of percentage. Chapter 5 explains the results of each model in detail, where diagonal elements are the correct prediction for each class.
There are 4 ways to check the predictions:
1.	The negative case were predicted negative
2.	The positive case were predicted positive
3.	The positive case were predicted negative
4.	The negative case were predicted positive.
Classification report comprises of precision,recall,f1 score and support.
Precision in the classification depicts the percentage of correct prediction.
Recall in the classification depicts the positive cases.
A macro-average first make the metric for each class independently and then take the average, whereas a micro-average takes avg of all class for calculating the metric.
F1 score tells the how many positives prediction were correct(1.0 best score and 0.0 worst)

2 Example of HVPS for VGG16 model which is having 145 test classes

Aggregate - 27
Bullet rosette - 40
Columnar crystal – 25
Compact particle – 26
Quasi-sphere - 27

The confusion matrix(Figure4.7) below shows that all the diagonal elements are the correct prediction for each class example for aggregate out of 27, 23 were predicted correctly true labels matches the predicted labels similarly for rest. This is the confusion matrix without normalisation, whereas the classification report(Figure 4.8) below shows the precision, recall and f1 score of each particle with aggregate showing correct prediction of 90% depicted positive cases that is recall 96% and 93% of the aggregate positive values are predicted correctly.



