# Lightweight-Modular-Model (LMM)

[![DOI](https://zenodo.org/badge/731119275.svg)](https://zenodo.org/doi/10.5281/zenodo.11166210)

This is the official implementation of the proposed model in the paper "Lightweight Yet Effective: A Modular Approach to Crack Segmentation".

The model was trained and tested on five datasets "Crack500, DeepCrack, GAPs384, AigleRN-TRIMM, and ShadowCrack". The datasets in this repository are NOT the original datasets but a cropped version, which we cropped using the code in the file `image_crop`. If you want to use the original datasets, please refer to the links below:

1. Crack500: (https://www.kaggle.com/datasets/pauldavid22/crack50020220509t090436z001?select=CRACK500)
2. DeepCrack: (https://github.com/yhlleo/DeepCrack/tree/master)
3. GAPs384: (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGIEBY)
4. AigleRN-TRIMM: (https://www.irit.fr/~Sylvie.Chambon/Crack_Detection_Database.html)
5. ShadowCrack: (lilifan@bit.edu.cn)

![alt text](https://github.com/Omaralmaqtari/Lightweight-Modular-Model/blob/main/Model%20Architecture.png?raw=true)

## Requirements
1. python 3.11.7
2. pytorch 2.2.0
3. opencv 4.9.0
4. numpy 1.26.4
5. scipy 1.11.4
6. matplotlib 3.8.0

## Cite this article
O. Al-maqtari, B. Peng, Z. Al-Huda, A. Al-Malahi and N. Maqtary, "Lightweight Yet Effective: A Modular Approach to Crack Segmentation," in IEEE Transactions on Intelligent Vehicles, https://doi.org/10.1109/TIV.2024.3405495.
