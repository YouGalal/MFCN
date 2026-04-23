Automated Precision Localization of Peripherally Inserted Central Catheter Tip through Model-Agnostic Multi-Stage Networks

If you want to use this repository in a practical setting, as in uploading your own x-ray images and running inference without corresponding ground truth segmentation masks, please apply the following modifications to the repository first.

- update "/MFCN/MFCN_SMC/model_input/input_Catheter_Whole_RANZCR" with your own images. Leave "MFCN/MFCN_SMC/model_input/mask_Catheter_Whole_RANZCR" empty
- delete folder "/MFCN/MFCN_SMC/output
- in "stage1_conventional_model.py" uncomment these lines: line 124, line 132
- in "stage2_patch_wise.py" comment out this line: line 204
- in "stage3_line_reconnection.py" comment out this line: line 195

<img width="841" height="838" alt="Screenshot 2025-10-28 182603" src="https://github.com/user-attachments/assets/35afe4d5-2ef7-46ba-9069-fbafc3b74282" />

<img width="1152" height="570" alt="Screenshot 2026-04-23 115339" src="https://github.com/user-attachments/assets/e798ec71-7eca-4cf2-9f6e-ee59a6623a26" />

<img width="1468" height="686" alt="Screenshot 2026-04-23 115438" src="https://github.com/user-attachments/assets/af2900ce-33bc-4000-aba7-26f3d885daff" />

<img width="899" height="855" alt="Screenshot 2025-10-28 200709" src="https://github.com/user-attachments/assets/c4d486bd-354e-4ac2-bd34-7a6a7f48ce84" />
