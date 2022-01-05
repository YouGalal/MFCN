## Automated Precision Localization of Peripherally Inserted Central Catheter Tip through Model-Agnostic Multi-Stage Networks

we release MFCN evaluation code.    
Collaborators: Kyung-su Kim, Myung Jin chung, Yoon Ki Cha, Subin Park   
Detailed instructions for testing the image are as follows.   


## MFCN Network
MFCN is Multi-stage Networks that improves PICC tip detection performance through multi-fragment phenomenon improvement.   
Multi fragment phenomenon (MFP) is a phenomenon in which some breaks occur in the predicted line when segmenting the sparse PICC area, making it difficult to accurately detect the catheter tip.   


Each model can be used Model-Agnostic, but it is a code that applies [FCDenseNet](https://arxiv.org/abs/1611.09326) with the best performance among the current popular segmentation models through experiments.    
MFCN consists of a total of three stages. The first stage is the conventional method, the second stage is the Patch-wise PICC segmentation network, and the third stage is the Line reconnection network that can directly solve the MFP.

![Figure2](https://user-images.githubusercontent.com/79253022/148063562-edbe9208-259b-4e59-807b-c9d59a9e20b2.jpg)

## Evaluation
#### Environments
the setting of the virtual environment we used is described as pytorch_MFCN.yml

#### PICC Datasets
We measured the experimental results using the RANZCR dataset, and the example image is in the `~/MFCN_SMC/model_input`.   
The RANZCR dataset can be downloaded from
<https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data>

#### pre-trained weight
segmentation model pre-trained weight file put it in `~/MFCN_SMC/checkpoint`.   
Download segmentation_checkpoint file in [here](https://drive.google.com/drive/folders/1p3RWyCzoQq8b4PWbgN_YNSNAqtSejcT-?usp=sharing)   

#### Conventional method
Please run "segmentation/codes/inference.py".
```
python.sh ## fixed
```

#### Patch-wise PICC segmentation network
```
python.sh ## fixed
```

#### Line reconnection network
```
python.sh ## fixed
```

## Result
if you run three things sequentially, you will see that a `~/MFCN_SMC/output/output_FCDenseNetmodel_`

![Figure14](https://user-images.githubusercontent.com/79253022/148207780-4a2d1fef-a6e0-46f5-a184-0285b0637377.jpg)

