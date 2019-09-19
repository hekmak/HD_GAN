# HD-GAN 
Tensorflow implementation 

# Introduction
HD GAN with Group norm, Hinge loss, ...

Generating persian miniature:

<img width="100" align="cener" src="miniature_0.gif">

## Installing
The code is tested with Cuda 9 and tensorflow 1.10 on Titan V.<br/>
We suggest using our docker to run the code.<br/>
(The docker includes Cuda 9 with tensorflow 1.10, ros kinetict and pcl library for point-cloud visualization.)
```
bash docker_run.sh
```
## Training
Inside the docker:
```
cd project/
python3 main.py --phase train
```


