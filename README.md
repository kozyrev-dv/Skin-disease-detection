# Skin disease detection using convolutional networks

The project is aimed towards illustraiting the difficulties in assessing the proper diagnosis of a skin and subcutaneous tissue diseases. Project uses 4 types of model architectures which where tested on ISIC dataset challenge 2019, consisting of 25331 training images and 8238 validation images. Treprocessing could be maid via *preprocessing.py* script and are stored in `.bin` files.

The tested models are:

- ResNet-18 convolution + SGD classifier: *"model_sgdc_512.p"*

- ResNet-18 convolution + 2-layer (1024, 218) feed-forward network: *"model_mlp_0.43_512.p"*

- EfficientNet-b5 + SGD classifier: *"model_sgdc_2048_efnet_b5.p"*

- EfficientNet-b5 + 2-layer (1024, 218) feed-forward network: *"model_mlp_0.39_2048.p"*
