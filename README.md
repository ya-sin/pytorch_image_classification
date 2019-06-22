# pytorch_image_classification
Image Classification implemented in PyTorch.
The dataset contains 28x28 (px) images of digits 0 to 8 in grayscale.Aim at recognizing the digit on the wooden block.
This repository includes:
1. self-collected data set
2. training and testing the model
3. a config.jason for setting
4. the net refer to utils/net.py
## Requirements
- torch
- torchvision
- numpy
- cv2
- PIL
- jason
## Dataset
![dataset](https://i.imgur.com/Gual5EB.png)

## Train Model
All the parameters are in **config.json**

```python train.py```
## Test Model
All the parameters are in **config.json**

```python test.py```
or
```python infer.py```
