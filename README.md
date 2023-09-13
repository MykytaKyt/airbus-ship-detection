# Airbus Ship Detection Challenge
The project contains my solution
of [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection/overview).

## Installation

Clone project

```$ git@github.com:MykytaKyt/airbus-ship-detection.git```

Then you need download model weights and unzip to root. Download by [link](https://drive.google.com/file/d/19CFruE4kWbvXzOKCOtTyzY4cCq7AgeMu/view?usp=drive_link)
### Install the requirements
Create conda env

```$ conda create --name py309 python==3.9```

Then activate it

```$ conda activate py309```

Run  ```$ pip install -r requirements.txt```

### Configure parameters 
The ```Makefile``` file contains configuration variables that you may need to adjust.

Also make sure that you installed CUDA toolkit and its corresponding cuDNN file to work with GPU.
Use Nvidia instructions to install it on your platform.

## Prepare dataset 
First you need to download the dataset if you want to train the model it can be found
at this [link](https://www.kaggle.com/c/airbus-ship-detection/data). Then unzip it to the `data` folder.
### EDA
The provided code in `notebooks/EDA.ipynb` performs exploratory data analysis (EDA) on a ship segmentation dataset, including analyzing
annotations, visualizing images with and without labels, and checking image existence and properties.
The results of the analysis are then saved as a preprocessed DataFrame in .parquet format to work with it in preprocessing.
### Preprocessing 
To start preprocess data you need use `utils/stages` and there `Makefile`.

Run command ```cd utils/stages```
#### Data Filtering
To filter the dataset, run the following command:

```make filter```

This will execute the `filter.py` script,
which filters the input dataset file and saves the filtered dataset to an output file.
It needs to remove images without masks.
#### Data Cleaning
To clean the filtered dataset, run the following command:

```make clean```

This will execute the `deduplication.py` script, which performs data deduplication
and other cleaning operations on the filtered dataset. It uses fastdub library, to find and remove blurry files,
broken files, duplicates, outliers, too bright and too dark images. 
#### Dataset Splitting
To split the cleaned dataset into train and test sets, run the following command:

```make split```

This will execute the `train_test_split.py` script, which splits the cleaned dataset into train and test sets based on
the provided test size. For train test split used fiftyone lib, to calculate uniqueness of every image
and then sort by it. Then the most unique images are put into the test set based on the split percentage.
#### Dataset Creation
To create the final dataset, run the following command:

```make create```

This will execute the `dataset.py` script, which creates a dataset by processing the split dataset and corresponding
image files. The final dataset will be saved in the specified output folder.
## Usage
For simple usage use `Makefile` in root dir.

### Train
For training, the U-Net model with a ResNet50 backbone is utilized.
This architecture combines the powerful feature extraction capabilities of ResNet50 with the precise segmentation 
ability of U-Net. As the loss function, the weighted binary cross-entropy and Dice loss are used.
This choice is made because the weighted BCE loss handles class imbalance by assigning different weights to
positive and negative classes, while the Dice loss encourages accurate segmentation by measuring the overlap between
predicted and ground truth masks. As metrics, the Dice coefficient and other standard segmentation metrics are used.
These metrics provide a comprehensive evaluation of the model's segmentation performance, considering both pixel-wise
classification accuracy and overall segmentation similarity.

To train model, you can modify training parameters in `Makefile`.
```
* save_dir: The directory where the trained model will be saved.
* save_name: The name of the saved model.
* logs: The directory where TensorBoard logs will be stored.
* epoch: The number of training epochs.
* batch_size: The batch size for training.
* lr: The learning rate for the optimizer.
* patience: The number of epochs to wait for improvement during early stopping.
* weight_decay: The weight decay for the optimizer.
* image_size: The desired image size for resizing.
* train_dir: The path to the directory containing the training data.
* test_dir: The path to the directory containing the testing data.
```
And then simply run 

`make train`

### Inference
To run inference script modify `Makefile` if you need.

And then run:

```make infer```

The script will perform inference on the input image using the specified model.
The generated segmentation mask will be saved as an output image at the specified output path.
Make sure you have the required files and directories set up correctly, including the trained model file and the input image file.
### Demo
You can use the demo to test the model.
Run the following command to launch the application:

```make demo```

Gradio will launch a user interface in your default web browser.

Use the file upload button in the user interface to select an image for segmentation.

The selected image will be preprocessed and segmented using the pre-trained model.
### API
Run the following command to launch the API:

```make api```

This command executes the api target in the Makefile.
The API server will start running, and the FastAPI framework will provide information about the routes and endpoints.

You can send a POST request to the /segment endpoint with an image file to perform image segmentation.

Make a POST request to `http://localhost:8000/segment` using your preferred API testing tool (e.g., curl, Postman).

Attach an image file in the request body.

The API will preprocess the input image, perform segmentation using the specified model, and return the segmented image.


#### Testing the API
To run the unit tests for the image segmentation API using the provided test_app.py script and Makefile,
run the following command to execute the unit tests:

```
make unit_tests
```
The unit tests will be executed, and the test results will be displayed in the command prompt or terminal.

* If all tests pass, you will see the output OK indicating that all assertions passed.
* If any tests fail, you will see the details of the failed tests and the reason for the failures.

To run the load test for the image segmentation API using the provided locust_load.py script and Makefile.
Run the following command to start the load test:

```make load_test```

The Locust load testing tool will start running, and the command prompt or terminal will display the Locust web interface [URL](http://localhost:8089).

Open a web browser and go to the provided [URL](http://localhost:8089) to access the Locust web interface.
Set the number of total users to simulate and the hatch rate (the number of users to spawn per second) in the web interface.
Click the "Start Swarming" button to start the load test.
Locust will start sending requests to the API server based on the defined tasks in the locust_load.py script.
You can monitor the load test progress and view statistics on the Locust web interface.

## Conclusion

In the "Airbus Ship Detection Challenge," the dataset presents several challenges that need to be addressed. Firstly, there is a limited dataset size (a big part of the images were without ships), which can lead to overfitting and hinder the model's generalization ability. Additionally, annotation errors, such as mislabeling or inaccurate boundary delineation, can introduce inconsistencies. Moreover, the class imbalance between ships and non-ship regions poses a difficulty for the model to effectively learn ship characteristics (compared to the non-ship region). Ships can vary significantly in terms of size, shape, orientation, and appearance. This variation can make it challenging for the model to accurately detect ships across different instances, especially if the training data does not cover the full range of variations. Lastly, limited diversity in environmental conditions and the presence of data artifacts and noise further impact the model's performance.

To solve this, I focused on preprocessing the data. I utilized techniques such as data augmentation to expand the dataset and reduce overfitting. Additionally, I applied cleaning of blurry, too-black, or too-white images, finding outliers and duplicates, to prevent overfitting. And take the most unique images for validation.

In terms of model selection, I thoroughly evaluated different architectures and settled on the Unet model with a ResNet50 backbone. This choice was driven by the need to segment small objects accurately, which a simple Unet architecture struggled with. The Unet model with a ResNet50 backbone showed promise in overcoming this challenge.

However, during training, I encountered some hurdles. I had some problems with detecting small-sized ships and overfitting. Despite the limitations, the model achieved a dice coefficient of 0.84 on the training set and 0.66 on the validation set. I recognized the need to mitigate the impact of waves and shadows through the implementation of a threshold-based fix. This adjustment improved the model's performance and could be further refined using tools like Gradio or inference.py, to test the model on difficult images. Maybe it is a good idea to make a test set with the most difficult images, but it needs more time and manual work.

Alternatively, you could try newer models or other approaches. For example, segment everything from Facebook. Or combine segmentation with something else.
Looking ahead, I believe a combination of detection, segmentation, and post-processing techniques would yield even better results. Integrating a detection model to identify ships and then applying segmentation to the cropped ship regions could enhance accuracy. However, due to time constraints, I was unable to implement this approach.
