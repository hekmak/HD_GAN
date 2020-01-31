# Periodontal prediction
Tensorflow implementation of the keypoints predictor for perio.
Healthy gums should fit snugly around each tooth, with the distance between the gum tissue and its attachment to the tooth only one to three millimeters in depth. 

# Introduction
ERFNet with IOU and sigmoid loss

## Installing
Use the docker to run the code (you might need ti change the project and dataset dirs in the run file).<br/> 
```
bash docker_run.sh
```

# Preprocessing
```
python3 pre_process.py
```
This will:<br/> 
1- Pad x-rays and masks.<br/> 
2- For each valid mask crops the corresponding tooth and creates perio keypoints.<br/> 
3- Saves the generated dataset.<br/> 

<img width="100" align="cener" src="./boz.jpg">
<img width="100" align="cener" src="boz_pad.jpg">
<img width="100" align="cener" src="tooth_img.jpg">
<img width="100" align="cener" src="tooth_label.jpg">
<img width="100" align="cener" src="tooth_mask.jpg">


## Training
```
cd project/
python3 main.py --phase train
```
Use tenssorboard for visualization.


