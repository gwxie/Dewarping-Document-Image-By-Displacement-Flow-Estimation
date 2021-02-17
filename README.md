# Dewarping Document Image
Dewarping Document Image By Displacement Flow Estimation with Fully Convolutional Network




# Dewarping Process
![image](https://github.com/gwxie/Dewarping-Document-Image-/blob/main/rectitify_image.jpg)
We predict the displacement and the categories (foreground or background) at pixellevel by applying two tasks in FCN, and then remove the background of the input
image, and mapped the foreground pixels to rectified image by interpolation according to the predicted displacements. The cracks maybe emerge in rectified image when using a forward mapping interpolation. Therefore, we construct Delaunay triangulations in all scattered pixels and then using interpolation.



# Release Code
1、Download model parameter  
2、Resize the input image into 1024x960 (zooming in or out along the longest side and keeping the aspect ration, then filling zero for padding. )  
3、Run "python test.py"  


# The complete code will be released soon.
