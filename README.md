# **Classification of Fashion MNIST using PyTorch**

## Context
Fashion-MNIST, a dataset launched by Zalando, provides a collection of article images that offers an exciting yet challenging alternative for training and testing machine learning models. The dataset comprises 70,000 grayscale images, divided into a training set of 60,000 examples and a test set of 10,000 examples. Each image is 28 pixels by 28 pixels, contained within a 28x28 matrix, where each pixel-value ranges from 0 to 255 indicating varying levels of lightness or darkness.

Through this project, I'm excited to deepen my hands-on experience with PyTorch in an image classification context and build a strong foundation for future projects that involve more complex datasets and advanced machine learning models.

## Prerequisites
The project uses multiple libraries used frequently in machine learning and deep learning projects.
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib
pip install seaborn
pip install numpy
pip install pandas
pip install scikit-learn
```

## Model Architecture
The CNN class defines a convolutional neural network architecture in PyTorch for classifying images. It starts with an input layer that assumes single-channel (grayscale) images. The first convolutional block consists of two layers of convolution, each with 32 filters of size 3x3, followed by ReLU activation and batch normalization for each convolution. This is followed by a 2x2 max pooling layer that reduces the spatial dimensions by half. The second convolutional block similarly has two convolution layers with 64 filters each, also followed by ReLU activation and batch normalization. This is followed by another 2x2 max pooling layer, further reducing the dimensionality. The network then flattens the output from the convolutional blocks and passes it through a fully connected layer with 512 units, followed by ReLU activation, and concludes with a linear layer that maps to the number of labels specified for classification. This architecture is designed to progressively extract more complex features from the input image, reduce spatial dimensions, and finally classify the image into one of the specified categories.

## Conclusions
The model managed to achieve a final accuracy of `92.54%` with a `0.2833` loss after 10 epochs of training with 64 batch size. 