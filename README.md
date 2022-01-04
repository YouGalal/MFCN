## Automated Precision Localization of Peripherally Inserted Central Catheter Tip through Model-Agnostic Multi-Stage Networks

we release MFCN evaluation code.    
Collaborators: Kyung-su Kim, Myung Jin chung, Yoon Ki Cha, Subin Park   
Detailed instructions for testing the image are as follows.   


MFCN is Multi-stage Networks that improves PICC tip detection performance through multi-fragment phenomenon improvement.   
Multi fragment phenomenon (MFP) is a phenomenon in which some breaks occur in the predicted line when segmenting the sparse PICC area, making it difficult to accurately detect the catheter tip.   

## MFCN Network
Each model can be used Model-Agnostic, but it is a code that applies FCDenseNet with the best performance among the current popular segmentation models through experiments.    
MFCN consists of a total of three stages. The first stage is the conventional method, the second stage is the Patch-wise PICC segmentation network, and the third stage is the Line reconnection network that can directly solve the MFP.

![Figure2](https://user-images.githubusercontent.com/79253022/148063562-edbe9208-259b-4e59-807b-c9d59a9e20b2.jpg)

## Environments
the setting of the virtual environment we used is described as pytorch_MFCN.yml

## PICC Datasets
We measured the experimental results using the RANZCR dataset, and the example image is in the `~/model_input`. 
The RANZCR dataset can be downloaded from
<https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data>
