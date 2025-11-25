# AML_Project

## Data description
The dataset contains bird images, divided into train and test splits. The images are inside test_images and train_images folders.

The labels of the training images are inside train_images.csv file. In this file, the first column is image_path and the second one is the label (1 - 200). The test_images_samples.csv includes a row id with a dummy label. The final goal of the challenge is to change the label column to the predicted label.

The class_names.npy is a dictionary including the name of each label. Load the file using the following code: np.load("class_names.npy", allow_pickle=True).item()

The structure of the final submission should be exactly the same as the test_images_samples.csv! Otherwise, it will fail.

Files
train_images - the training images
test_images - the test images
test_images_sample.csv - a sample submission file in the correct format
test_images_path.csv - path to test file images
train_images.csv - supplemental information about the data
class_names.npy - this file includes the name of each label
attributes.npy - this file includes the attributes which are extra information for each class.
attributes.txt - this file includes the attribute names which are extra information for each class.
