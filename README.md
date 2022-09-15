# Machine-learning-models
PRETRAINEND MODELS

https://github.com/aisha-colab/Machine-learning-models/tree/main
Use google colab for this work
Architecture of CNN 
The working of CNN on the images taken by two probes (HVPS and 2D-S)
4.1. Architecture of CNN: In image classification each pixel plays an important role in make correct prediction in normal feed forward system the conversion of two-dimensional image into single array and feeding it makes the image loss its importance so to overcome this drawback convolutional neural network comes into play as it provides convolutional layer along with pooling layer.


4.1.1. Convolutional layer: As the name suggest it operation called convolution in which there is feature extraction done by applying this operation at specific kernel and specific pattern. Specific patterns of the images such as edges, dimensions are detected by filters.
There are multiple filters in convolution layer and per filter gives on feature map as output. Relu activation ids applied after convolution in order put zero for all non-negative values and to introduce nonlinearity.

 
Figure 4.2: Convolutional layer working shown
 
4.2.2. Pooling layer: After convolutional layer this layer is used to extract most dominant features of the image either by doing Max pooling or avg pooling depending on the required of the image dataset.

4.3.3. Fully connected layer: After the above two layers there Is fully connected layer which is like feed forward neural network in which the output of both the above layers are flatten into 1D array.
PRETRAINED MODELS 
VGG16
VGG19
RestNet50
Inception-V3
EfficientNet
XCeption
