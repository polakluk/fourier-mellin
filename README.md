# Image Registration using Log-polar transformation, Phase correlation (Fourier-Mellin)

Implemented in Python 2.7.11. using OpenCV, Pandas and NumPy

## Used Materials

This project is based on paper "An Application of Fourier-Mellin Transform in Image Registration" written by Xiaoxin Guo, Zhiwen Xu, Yinan Lu, Yunjie Pang.

## How to run code

In command-line, use command:
```
python script.py {original_image} {image_we_want_to_detect}
```
Example
```
python .\script.py .\test-data\lena_orig.png '.\test-data\lena_scale(90)_rot30.png'
```

## Noise

The script allows you to add two kinds of noise - Gaussian Noise and Salt & Pepper. The flag for turning one of the noises is at line 16
```
noiseMode = "none" # "gaussian", "s&p", "none"
```

At line 17, you can specify parameters for each type of noise:
```
noiseIntensity = {'sigma' : 2, 'mean' : 0, 'whiteThreshold' : 0.01, 'blackThreshold' : 0.99}
```
