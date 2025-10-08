Automated Precision Localization of Peripherally Inserted Central Catheter Tip through Model-Agnostic Multi-Stage Networks

If you want to use this repository in a practical setting, as in uploading your own x-ray images and running inference without corresponding ground truth segmentation masks, please apply the following modifications to the repository first.

- update "/MFCN/MFCN_SMC/model_input/input_Catheter_Whole_RANZCR" with your own images. Leave "MFCN/MFCN_SMC/model_input/mask_Catheter_Whole_RANZCR" empty
- delete folder "/MFCN/MFCN_SMC/output
- in "stage1_conventional_model.py" uncomment these lines: line 124, line 132
- in "stage2_patch_wise.py" comment out this line: line 204
- in "stage3_line_reconnection.py" comment out this line: line 195
