# GluonCV-Torch

Load [GluonCV](https://gluon-cv.mxnet.io/) Models in PyTorch.
Simply `import gluoncvth` to getting better pretrained model than `torchvision`:

```python
import gluoncvth as gcv
model = gcv.models.resnet50(pretrained=True)
```

Installation:

```bash
pip install gluoncv-torch
```


## What's in the package?

### ImageNet Models

ImageNet models single-crop error rates, comparing to the `torchvision` models:

|                                 | torchvision     |               | gluoncvth     |             |
|---------------------------------|-----------------|---------------|---------------|-------------|
| Model                           | Top-1 error     | Top-5 error   | Top-1 error   | Top-5 error |  
| ResNet18                        | 30.24           | 10.92         | 29.06         | 10.17       |
| ResNet34                        | 26.70           | 8.58          | 25.35         | 7.92        |
| ResNet50                        | 23.85           | 7.13          | 22.33         | 6.18        |
| ResNet101                       | 22.63           | 6.44          | **20.80**     | **5.39**    |
| ResNet-152                      | 21.69           | 5.94          | 20.56         | 5.39        |
| Inception v3                    | 22.55           | 6.44          | 21.33         | 5.61        |

More models available at [GluonCV Image Classification ModelZoo](https://gluon-cv.mxnet.io/model_zoo/classification.html#imagenet)

### Semantic Segmentation Models

Results on ADE20K dataset:

| Model         | Base Network  | PixAcc    | mIoU       |
|---------------|---------------|-----------|------------|
| FCN           | ResNet101     | 80.6      | 41.6       |
| PSPNet        | ResNet101     | 80.8      | 42.9       |
| DeepLab       | ResNet101     | 81.1      | 44.1       |

Results on Pascal VOC dataset:

| Model         | Base Network  | mIoU       |
|---------------|---------------|------------|
| FCN           | ResNet101     | 83.6       |
| PSPNet        | ResNet101     | 85.1       |
| DeepLab       | ResNet101     | 86.2       |

More models available at [GluonCV Semantic Segmentation ModelZoo](https://gluon-cv.mxnet.io/model_zoo/segmentation.html)

## Why [GluonCV](https://gluon-cv.mxnet.io/)?

**1. State-of-the-art Implementations**

**2. Pretrained Models and Tutorials**

**3. Community Support**

We expect this PyTorch inference API for GluonCV models will be beneficial to the entire computer vision comunity.
