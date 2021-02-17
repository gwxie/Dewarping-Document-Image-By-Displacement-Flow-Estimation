# Dewarping Document Image
Dewarping Document Image By Displacement Flow Estimation with Fully Convolutional Network




# Dewarping Process
![image](https://github.com/gwxie/Dewarping-Document-Image-/blob/main/rectitify_image.jpg)
We predict the displacement and the categories (foreground or background) at pixellevel by applying two tasks in FCN, and then remove the background of the input
image, and mapped the foreground pixels to rectified image by interpolation according to the predicted displacements. The cracks maybe emerge in rectified image when using a forward mapping interpolation. Therefore, we construct Delaunay triangulations in all scattered pixels and then using interpolation.




# The complete code will be released soon.
