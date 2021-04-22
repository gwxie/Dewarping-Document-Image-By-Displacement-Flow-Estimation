# Description
"2019-06-25 11:52:54/49": model parameter. Pre-trained models are available [here](https://drive.google.com/file/d/17MtuXcgMqb5HjBRy4tNL9oLkEI5auLUT/view?usp=sharing).

"dataset/shrink_1024_960/crop": test dataset. We resized the image into 1024x960 (zooming in or out along the longest side and keeping the aspect ration, then filling zero for padding. )

"dataloader.py": load data. 

"loss.py": loss function. 

"network.py": network structure.

"test.py": main program.

"train.py": train model:.

"utils.py": post-processing.



# Running
1、Download model parameter and source codes 

2、Resize the input image into 1024x960 (zooming in or out along the longest side and keeping the aspect ration, then filling zero for padding. )  

3、Run `python test.py --data_path_test=./dataset/shrink_1024_960/crop/`

# Training
Run `python train.py --data_path_train=./dataset/unwarp_new/train/data1024_greyV2/color/  --data_path_validate=./dataset/unwarp_new/train/data1024_greyV2/color/ --data_path_test=./dataset/shrink_1024_960/crop/`


