# Cross-Modal Self-Attention Network for Referring Image Segmentation

This repository contains code and trained model for the paper ["Cross-Modal Self-Attention Network for Referring Image Segmentation"](https://arxiv.org/abs/1904.04745), CVPR 2019.

If you find this code or pre-trained models useful, please cite the following papers:
```
@inproceedings{ye2019cross,
  title={Cross-Modal Self-Attention Network for Referring Image Segmentation},
  author={Ye, Linwei and Rochan, Mrigank and Liu, Zhi and Wang, Yang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={10502--10511},
  year={2019}
}
```
## Requirement
- Python 2.7
- Tensorflow 1.2 or higher
- [PyDenseCRF](https://github.com/lucasb-eyer/pydensecrf)

## Setup
Partial coda and data preparation are borrowed from [TF-phrasecut-public](https://github.com/chenxi116/TF-phrasecut-public). Please follow their instructions to make your setup ready. DeepLab backbone network is based on [tensorflow-deeplab-resnet](https://github.com/DrSleep/tensorflow-deeplab-resnet) as well as the pretrained model for initializing weights of our model. 

## Sample code
### Training
```
python main_cmsa.py -m train -w deeplab -d Gref -t train -g 0 -i 800000
```


### Testing 
```
python main_cmsa.py -m test -w deeplab -d Gref -t val -g 0 -i 800000
```
A trained model is available [here](https://drive.google.com/open?id=1LdqQnOozO9553MIl14gwglkBXWIXXUzz). You should be able to produce results on Gref validation dataset as 39.96% / 40.07% (without/with CRF) in terms of IoU.
