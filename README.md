# Dewarping Document Image
Dewarping Document Image By Displacement Flow Estimation with Fully Convolutional Network.

Please browse 90_paper.pdf




# Dewarping Process
![image](https://github.com/gwxie/Dewarping-Document-Image-/blob/main/rectitify_image.jpg)
We predict the displacement and the categories (foreground or background) at pixellevel by applying two tasks in FCN, and then remove the background of the input
image, and mapped the foreground pixels to rectified image by interpolation according to the predicted displacements. The cracks maybe emerge in rectified image when using a forward mapping interpolation. Therefore, we construct Delaunay triangulations in all scattered pixels and then using interpolation.


# Notice
2020.11.10 update the result file, such as 6-25_11_52_54-49-rgb_ and 6-25_11_52_54-49_.

2022.2.17 update the Release Code.


# Release Code
The source code is open, please download from Source. 

Sorry to be busy with master's thesis and graduation matters. Please understand that the reply is slow.

Please send an email to xieguowang2018@ia.ac.cn.

# Running
1、Download model parameter and source codes 

2、Resize the input image into 1024x960 (zooming in or out along the longest side and keeping the aspect ration, then filling zero for padding. )  

3、Run "python test.py"  


# The complete code will be released soon.
