
# Adaptive Parametric Prototype Learning for Cross-Domain Few-Shot Classification

PyTorch implementation of:
 Adaptive Parametric Prototype Learning for Cross-Domain Few-Shot Classification


## Datasets
The following datasets are used for evaluation in this challenge:

### Source domain: 

* miniImageNet 


### Target domains:
We use miniImageNet as the single source domain, and use CUB, Cars, Places, Plantae, CropDiseases, EuroSAT, ISIC and ChestX as the target domains.

For miniImageNet, CUB, Cars, Places and Plantae, download and process them seperately with the following commands.
- Set `DATASET_NAME` to: `miniImagenet`, `cub`, `cars`, `places` or `plantae`.
```
cd filelists
python process.py DATASET_NAME
cd ..
```

For CropDiseases, EuroSAT, ISIC and ChestX, download them from

* **EuroSAT**:

    Home: http://madm.dfki.de/downloads

    Direct: http://madm.dfki.de/files/sentinel/EuroSAT.zip

* **ISIC2018**:

    Home: http://challenge2018.isic-archive.com

    Direct (must login): https://challenge.isic-archive.com/data#2018

* **Plant Disease**:

    Home: https://www.kaggle.com/saroz014/plant-disease/

    Direct: command line `kaggle datasets download -d plant-disease/data`

* **ChestX-Ray8**:

    Home: https://www.kaggle.com/nih-chest-xrays/data

    Direct: command line `kaggle datasets download -d nih-chest-xrays/data`


## Steps

1. Download the datasets for evaluation (EuroSAT, ISIC2018, Plant Disease, ChestX-Ray8) using the above links.

2. Download miniImageNet using <https://drive.google.com/file/d/1uxpnJ3Pmmwl-6779qiVJ5JpWwOGl48xt/view?usp=sharing>


4. Change configuration file `./configs.py` to reflect the correct paths to each dataset. Please see the existing example paths for information on which subfolders these paths should point to.

5. Train base models on miniImageNet

    • *Standard supervised learning on miniImageNet*

    ```bash
        python ./train.py --dataset miniImageNet --model WideResNet28_10   --method baseline --train_aug
    ```

    • *Train meta-learning method (protonet) on miniImageNet*

    ```bash
        python ./train.py --dataset miniImageNet --model WideResNet28_10   --method protonet --n_shot 5 --train_aug
    ```

6. Save features for evaluation (optional, if there is no need to adapt the features during testing) 

    • *Save features for testing*

    ```bash
        python save_features.py --model WideResNet28_10  --method protonet --dataset CropDisease --n_shot 5 --train_aug
    ```

7. Test with saved features (optional, if there is no need to adapt the features during testing) 

    ```bash
        python test_with_saved_features.py --model ResNet10 --method protonet --dataset CropDisease --n_shot 5 --train_aug
    ```

8. Test
    • *Finetune*

    ```bash
        python finetune.py --model WideResNet28_10  --method protonet  --train_aug --n_shot 5 
    ```
    
    • *Example output:* 600 Test Acc = 49.91% +- 0.44%



