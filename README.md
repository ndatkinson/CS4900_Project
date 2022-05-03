# CS4900_Project
Group Project for Senior Seminar



###############################################
-----------------------------------------------
____Python Script for Training a CNN Model_____
+++++++++++++++++++++++++++++++++++++++++++++++
The Python script consists of several .py files. Train.py is run to train the model and calls the other .py files as part of the training process.
The model will be created from the .py files and used in the Android component of this project. The script can be run to produce the model used in the image segmentation part of this project. 
Here are the steps to take to produce the model we used
1. Arrange the files in their directories in the project directory for the script. The train and predict files should go in the root directory and there should be a Dataset, output, and pyimageseach folders in the project directory.
The model.py, config.py, and dataset.py files should go in the pyimagesearch folder.
2. Run the train.py file in terminal or command prompt to produce the model.
3. Android Studio Instructions: launch AS and cold boot emulator by opening the drop down menu in the device manager, and choose Cold Boot. Do NOT press the green run button that is in the device manager: it causes the emulator to freeze up for some reason, and the storage will have to be wiped and everything put back in. After the device has booted, press the green run button in Android Studio to run the program on the emulator. In the app, choose Load to pick an image from the gallery. After selecting the desired image, press the segment button, which will display the appropriate segmentation mask after the model has been run on the image.
+++++++++++++++++++++++++++++++++++++++++++++++
_________________End of Instructions___________________
###############################################
