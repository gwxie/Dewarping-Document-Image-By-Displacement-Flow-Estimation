# Dewarping Document Image
Dewarping Document Image By Displacement Flow Estimation with Fully Convolutional Network.


See [“paper”](https://arxiv.org/abs/2104.06815) for more information.

Download [“rectified images”](https://drive.google.com/drive/folders/1sG0zi1Y0DQSg0VGeo-c_wKU1TVgcvNHp?usp=sharing).

# Dewarping Process
![image](https://github.com/gwxie/Dewarping-Document-Image-By-Displacement-Flow-Estimation/blob/main/rectitify_image.jpg)
We predict the displacement and the categories (foreground or background) at pixellevel by applying two tasks in FCN, and then remove the background of the input
image, and mapped the foreground pixels to rectified image by interpolation according to the predicted displacements. The cracks maybe emerge in rectified image when using a forward mapping interpolation. Therefore, we construct Delaunay triangulations in all scattered pixels and then using interpolation.

# Compare
![image](https://github.com/gwxie/Dewarping-Document-Image-By-Displacement-Flow-Estimation/blob/main/compare.jpg)

# Requirements
<p>python >=3.7</p>
<p>pytorch</p>
<p>opencv-python</p>
<p>scipy</p>

# Notice
- 2020.11.10 update the result file, including 6-25_11_52_54-49-rgb_ and 6-25_11_52_54-49_.

- 2022.2.17 update the Release Code.

- 2022.4.14 update Source file.


# Release Code
The source code is open, please download from Source. 

Please send an email to gwxie@outlook.com.

# Running
1、Download model parameter and source codes 

2、Resize the input image into 1024x960 (zooming in or out along the longest side and keeping the aspect ration, then filling zero for padding. )  

3、Run `python test.py --data_path_test=./dataset/shrink_1024_960/crop/`

# Training
Run `python train.py`

# Dataset
The training dataset can be synthesised using the [scripts](https://github.com/gwxie/Distorted-Image-With-Flow).

