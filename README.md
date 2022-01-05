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
Currently, all code examples are assuming distributed launch with 2 multi GPUs.   
the setting of the virtual environment we used is described as `~/pytorch_MFCN.yml`

#### PICC Datasets
We measured the experimental results using the RANZCR dataset, and the example image is in the `~/MFCN_SMC/model_input`.   
The RANZCR dataset can be downloaded from
<https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data>

#### pre-trained weight
segmentation model pre-trained weight file put it in `~/MFCN_SMC/checkpoint`.   
Download segmentation_checkpoint file in [here](https://drive.google.com/drive/folders/1p3RWyCzoQq8b4PWbgN_YNSNAqtSejcT-?usp=sharing)   

#### Conventional method
To evaluate First stage, run this command:
```
python stage1_conventional_model.py
```

#### Patch-wise PICC segmentation network
To evaluate Second stage, run this command:
```
python stage2_patch_wise.py
```

#### Line reconnection network
To evaluate First stage, run this command:
```
python stage3_line_reconnection.py
```


## Result

Our model achieve the following RMSE performance (mm) on:   

|Method|Conventional model|     MFCN     |
|------|---|---|
|FCDenseNet|95.90|43.06|
|U-Net|263.70|71.90|
|Attention U-Net|187.98|56.64|

If you run three things sequentially, You can check the final result in `~/MFCN_SMC/output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/FCDenseNetthird_output/FCDenseNet/First_connected_component`.    


The results of the conventional model can be found in `~/MFCN_SMC/output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/First_output/First_connected_component`.    
