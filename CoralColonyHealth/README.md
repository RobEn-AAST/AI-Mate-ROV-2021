# Task 2.2

Using image recognition to determine the health of a coral colony by comparing its current to a picture taken in the past

## Requirements

- open cv v3 or above.
- numpy.
- python version 3.6.8 is highly recommended. 
- Download PVC inside Detector/Weights, find link there
- Cuda, tested with cuda 11.2.152 on ubuntu

### NEED TO RECOMPILE DARKNET AT FIRST USE, #> cd Detector & make
### ~~Need to compile opencv * might be optional~~- OPTIONAL
however, Follow this tutorial to build opencv on ubuntu distros : https://linuxize.com/post/how-to-install-opencv-on-ubuntu-20-04/
and of course set OPENCV=1 in MakeFile dear.

## Installation


1 - using conda installation to install the virtual enviroment
```bash
conda create --name <enviroment name> --file requirments\\requirements.yml
```
2 -  by using the requirements.txt file in your created virtual enviroment add the needed libraries ( open-cv ) 

```bash
pip install -r requirments\\requirements.txt
```

## Usage

1- put the old image in the same directory and name it old.png

2- open cmd and type the following command
```bash
python <script directory>/task_2-2.py
```

## Theory

1- the back ground detection model is updated.

2- the background of the image is extracted using background removing object.

3- the frames are compared to the reference picture (old.png).

4- after comparison the changes detected are highlighted using contours.
