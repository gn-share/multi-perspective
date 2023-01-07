# Multi-perspective
This is the source code for the paper "Blind image quality assessment via multi-perspective consistency". Thank you for checking our code. I hope our code is useful to you.

# Dependencies

Before executing the code, you need to check whether the following python libraries are installed.

```
Python 3.6+
PyTorch 0.4+
TorchVision
scipy
csv
openpyxl
```

# Image Database Path
In train_test_IQA.py file, you need to change image path.
```
    folder_path = {
        'live':   '/image_data/LIVE/',  #
        'csiq':   '/image_data/CSIQ/',  #
        'tid2013':   '/image_data/tid2013',
        'livec':  '/image_data/ChallengeDB_release/',  #
        'koniq':   '/image_data/koniq/',  #
        'bid':   '/image_data/BID/',  #
    }
```


# Train and test on IQA database
Training and testing on LIVE database. 
```
python train_test_IQA.py
```

Some available options:
- ```--dataset```: Training and testing dataset, support datasets: livec | koniq-10k | bid | live | csiq | tid2013.
- ```--train_patch_num```: Sampled image patch number per training image.
- ```--test_patch_num```: Sampled image patch number per testing image.
- ```--batch_size```: Batch Size
- ```--model1```: The model1 is main model. Training and testing model1, support models: res18|res50|googlenet|vgg16
- ```--model2```: The model2 is assistant model. Training and testing model2, support models: res18|res50|googlenet|vgg16



