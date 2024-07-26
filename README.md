# HSMix


Pytorch codes of following paper:


HSMix: Hard and Soft Mixing Data Augmentation for Medical Image Segmentation



We propose HSMix, a novel approach to local image editing data augmentation involving hard and soft mixing for medical semantic segmentation. In our approach, a hard-augmented image is created by combining homogeneous regions (superpixels) from two source images. A soft mixing method further adjusts the brightness of these composed regions with brightness mixing based on locally aggregated pixel-wise saliency coefficients. The ground-truth segmentation masks of the two source images undergo the same mixing operations to generate the associated masks for the augmented images.
Our method fully exploits both the prior contour and saliency information, thus preserving local semantic information in the augmented images while enriching the augmentation space with more diversity. Our method is a plug-and-play solution that is model agnostic and applicable to a range of medical imaging modalities.


We utilize the combination of hard mixing and soft mixing.

The Hard Mixing can be illustrated in  the following figure.

![image](https://github.com/DanielaPlusPlus/HSMix/blob/main/Hard_Mixing.png)


The Soft Mixing can be illustrated in  the following figure.

![image](https://github.com/DanielaPlusPlus/HSMix/blob/main/Soft_Mixing.png)


The details of the related models and datasets can be found in 

https://github.com/DanielaPlusPlus/DataAug4Medical
