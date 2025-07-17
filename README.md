# PyTorch-GAN
Simpler, PyTorch implementation of a Generative Adversarial Network trained for image upscaling, <a href="https://github.com/milesd123/numpy-GAN">numpy-GAN</a>. Used to train <a href="https://huggingface.co/Milesd123/minecraft_swords_1m">minecraft sword file upscaler</a> on the dataset <a href="">16x_to_64x</a>. The model passes the alpha channel of an image and predicts the alpha channel of the label image, since minecraft swords files (PNG, JPEG) rely on RGBA, where only a similar portion of the image is filled. Combined with a model to color the alpha image, this could be a fun tool to upscale 16x minecraft sword images to 64. 

|Model | Generator |
|------|-----|
|# Trainable Parameters| ~ 1M|
|# Training Examples|~320,000|
|# Epochs |30 Total|
|Batch Size|32|
|Optimizer|Adam|
|Loss|L1, edge(*10), entropy(*2.5)|
