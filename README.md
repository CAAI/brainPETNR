# Brain PET Noise Reduction (brainPETNR)

Code referenced in Raphael S. Daveau, *et al.* *Deep learning based low-activity PET reconstruction of [18F]FE-PE2I and [11C]PiB in Neurodegenerative Disorders* (2022), currently in the submission process.

Code in **postprocessing** was used for:
- image inference
- image similarity metrics computation
- segmentations of ROIs
- clinical metrics computation

# Model

The code used for training is found at https://github.com/CAAI/rh-torch and the model is in rhtorch/models/autoencoders.py > AttentionUNet() (TODO: NEEDS TO BE PUSHED). 

The original inspiration of the code comes from https://github.com/momo1689/FAN