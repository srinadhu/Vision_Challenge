# Vision_Challenge
Please note that labels in .mat files are from 1 to 196 but in all the models are taken as from 0 to 195. In all these experiments the test set from the available dataset is used as validation set for experiments. The accuracy reported is all on the test set from available dataset.

Model 1: Data augmentation is resizing, randomcrop and horizontalflip. Data normalization is with mean and std of ImageNet. ResNet50 pre-trained model is used. The whole model is tuned. No weight decay and adam optimizer is used, ReducelrOnPlateau scheduler is used. The idea behind this approach is to see how well a simple baseline performs. The best model gave accuracy of 82.53%

Model 2: The model 1 was overfitting. Same as Model 1 but only last layer of ResNet50 is trained. This performs far worse giving the best accuracy of 53.32%. So sticking to full tuning of ResNet50 is a nice option. 

Model 3: Same as Model 1 but to overcome the overfitting problem added weight decay to the model. The accuracy did improve a bit to 83.98% compared to Model 1. Still the model was overfitting.

Model 4: Observed from the data that some of the images are slightly rotated. So added an data augmentation of rotation. It didn't perform great, the accuracy is 78.18%. The reason might be random-rotation and random-crop are cancelling each other. 

Model 5: There is information of both bounding box in train and test dataset. Instead of random crop used the crop from this bounding box to improve the model. performed experiments with weight decay. The best model accuracy is 86.10%. This is the best model I could achieve.

The test.py can be used for testing on new dataset, please make the necessary changes as mentioned in the file. The resnet50_adam_weights.pt is the best model weights. result.csv has the file name and label for the testset files after running test.py file. The test, train files and labels are in .mat files accordingly. The train.py is the training file and data_loader.py is the data loading file.  

