# Kaggle_Competition_Winter2021

MAIS 202 - Kaggle Competition Winter 2021 - Write Up 

1) Implementation of the model:

  I started by downloading the Kaggle dataset and my first step consisted of unzipping the train_x.npy and test_x.npy that I had stored in a folder on my google drive.
  I then proceeded in loading the images from train_x.npy  into a numpy array called train_images, and loaded the labels without the IDs from train_y.csv into a numpy array called train_labels. I repeated the process for the test data, but created an empty array for the test_labels since that is what the machine learning model will be predicting.
  I then transformed the images given in train_x.npy into PIL Images, then into Pytocrh tensors and then I normalized them.
  After that, I created the PlainDataset that I named LeaClassifier. I defined the __init__(), __len__()  and __getitem__() functions. For my __getitem__(), i passed the training labels as self.y and passed the training images as self.X and reshaped them to be 128x128 since that was the size of the input image.
  I then called DataLoader() on the train_data and the test_data, which are both instances of my LeaClassifier() class and this loaded all the labels and images for the training and the testing.
  I then created a CNN with 3 convolutional layers, 3 max-pooling functions and one activation function. For the activation function nn.Linear(3920,10), i set the output as 10 because the MNIST has 10 possibilities from 0 to 9. I also set the input to 3920 after running my code with a random number first, and then an error message told me it couldn’t map 3920 to the number i put. This is how I knew the right number was 3920.
  I finally wrote the train and testing method in which I used an optimizer to set the gradients to zero before starting to do backpropagation. This is due to the fact that because PyTorch accumulates the gradients on subsequent backward passes.

2)Results (how your model performed with various hyperparameters + your best model for the Kaggle competition)

The first time I created my CNN, I created 3 convolutional layers with 32, then 64 channels. I set my epoch to 20, and google colab took one hour and forty minutes to run 7 epochs then displayed “runtime disconnected”. 

Since it was taking me 30minutes for 2 epochs, I decided to change the number of channels to 10 as in the picture below, and i set the learning rate to 0.005 instead of the previous 0.001. The training sped up and each epoch took around 5 minutes to run but the model stagnated at 1.887 for loss.

I then set the channels to 10 and the learning rate back to 0.001. The training speed was faster because I could see more epochs in less time, but the error loss from one epoch to another was extremely small.
Although I know that until now the best training results were with my initial implementation (32 then 64 then 20 channels) with the lowest recorded loss of 0.889, I kept getting the following error when I tested my model (picture 4 above) and ran out of time, so submitted my code and write up as is.

3) Challenges (what was hard about the challenge/implementation):

At first, I didn’t know how to get the dataset onto my colab notebook, and then I followed these instructions that involved creating a google drive folder. https://medium.com/analytics-vidhya/how-to-fetch-kaggle-datasets-into-google-colab-ea682569851a. Then, I had a hard time opening the csv file because I kept calling read_csv() on it, but William(TPM) explained that I needed to call it by the path. It worked after I did that.
I also was unsure of how many maxpooling and activation functions to use, but then I watched the following youtube video and estimated from what I understood from it: https://www.youtube.com/watch?v=LgFNRIFxuUo&ab_channel=SungKim.
Finally, the way I build my training function was not working, and I found this resource that explained a step by step approach: https://debuggercafe.com/custom-dataset-and-dataloader-in-pytorch/. This really helped me understand what was happening and how all parts of my code connected together.

4) Conclusion (what did you learn from this assignment):

This assignment taught me how to create a PlainDataset Class, how to used DataLoaders and how to build a CNN. I also learned how to pick the number of channels in the different convolutional layers of my CNN and the concept of backpropagation for gradients in Pytorch.  



