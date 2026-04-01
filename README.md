# Odin

This Project has been developed by:
```
Sven Thorkildsen          Robotics Engineer @ Kameleon Solutions          sven.thorkildsen@kameleongruppen.no         
```


## Project description
The aim of this project is to develop a solution for training and deploying computer vision models for detecting defective objects during production, based on a set of images of approved products.


## User manual

### Gather data
#### Training data
The dataset used to train the model should contain ~20 images of products with an accepted production quality.
It is important that the images are representative for how the model will evaluate the products later, meaning that conditions like lighting, background and camera angle should be as similar as possible.

#### Test data
In order to verify that the model works as intended, the dataset used to test the model should contain every possible case, from accepted products to edge case defects.
There are no limitations to how big this set can be as it will only evaluate the model's performance, not change its behaviour.

### Upload data sets
Organize the datasets for training and test images in a folder like this:

```markdown
  - name_of_product
    * training
      - image_train_1
      - ...
      - image_20
    * test
      - image_test_1
      - ...
```

Give the folder a descriptive name and save it to the repository's **datasets** folder.

### Train model
TODO: HOW TO RUN SCRIPT FOR TRAINING MODEL

### Deploy model
TODO: HOW TO RUN SCRIPT FOR RUNNING LIVE DETECTION
#### Locally
TODO: RUN CODE ON TRAINING COMPUTER
#### Remote 
TODO: DEPLOY THE PROGRAM TO NVIDIA JETSON ORIN NANO



