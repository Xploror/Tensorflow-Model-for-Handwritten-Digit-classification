# Tensorflow-Model-for-Handwritten-Digit-classification
Using basic Keras layers to train a custom model using MNIST dataset

## Approach taken
1. Web scraping the handwritten digits from different website and storing it in corresponding numbered folders.
2. Making the dataset ready to be input in the model.
3. Making the custom model using Keras layers typically making an effective CNN model.
4. Training the model and finally testing it.

-----

There is an IPYNB file, 'Handwritten_digit_detection.ipynb' already having the combination of all teh steps from making of dataset till testing the model. Data Augmentation can be done seperately before inputting it to the model for which data_augmentating_code.py file is been given. Also an important note if following the notebook is that all the digit datasets were saved in respective numbered folders on the google drive and so uploading of google drive is important similarly MNIST dataset as csv file is also saved on google drive.

![2021-06-23 (3)](https://user-images.githubusercontent.com/69386934/123126039-af6f6d00-d466-11eb-8a6e-f1fa11e32f77.png)


The Project has been tested and trained for both the web-scraped images as well as MNIST dataset provided by Kaggle and thats why you will see two code cells for making final dataset.

  ![2021-06-23 (1)](https://user-images.githubusercontent.com/69386934/123119664-5bae5500-d461-11eb-815e-d18f83ad16b3.png)

Though its recommended to use MNIST dataset over web-scraped dataset as it contains 50,000 unique images of handwritten digits as compared to few thousands of images from web-scraping. Still if you are interested in making your own custom dataset just make the dataset as large as possible keeping in mind no redundancy in images or else your model may get overfit.

---

## Custom Model

```python
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, BatchNormalization, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
```
These imports are necessary to build your CNN model as well as fit it for the training dataset. For making your own custom model, Model is been sequenced with keras layers somewhat similar to the image below. Batch normalization layers are used to equally scale the outputs of the Convolutional layer which helps in regularizing as well as speeding up the training process.

![image](https://user-images.githubusercontent.com/69386934/123127497-f5790080-d467-11eb-8a61-8fc842cc0a58.png)
Image Reference - https://www.researchgate.net/figure/Architecture-of-the-tested-model-BaN-Batch-Normalization_fig2_338676025 

After building the model, it is made fit on the training dataset with labels taking bach_size=256, validation_split=0.2 for 5 epochs, but all these hyperparameter are flexible and may change for different model or even different dataset taken for optimal training. 

![2021-06-23 (4)](https://user-images.githubusercontent.com/69386934/123129576-adf37400-d469-11eb-8642-4b5f5ea09260.png)

As it is clear from the picture that the model's training accuracy is quite high but the validation accuracy is also fair enough by 97.73% accurate. Its clear that the model is a little bit overfit because the model was last trained using web scraped images which didnt had variety of images but still it performs quite well during testing.

----
## Testing section

![2021-06-23 (5)](https://user-images.githubusercontent.com/69386934/123131021-eba4cc80-d46a-11eb-867c-633cc783087d.png)

In this section, one image at a time has been taken to predict its value as seen above. First the test photo is read using imread in grayscale, then its resized to (28,28) as the model requires this input size. Then finally model.predict() is used to predict the class of the image giving a vector of size of number of classes. This vector contains the probability of the image on each class and the highest probability is the class predicted for the test image.

Here in the screenshot attached we can see that the test image got a prediction of digit 7 with 99.07% confidence.

