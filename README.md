# Generative-model-most-similar-images

After generating fake images with a Generator (generally GANs), many are concerned about whether the generated images are really novel or rather just copies of the training dataset. This repository contains a python function that output the 5 most similar training (real) images to the given generated (fake) images. I couldn't find an implementation for this so I made one. It does not require any deep learning library and you only need the training images and the fake generated images (of batch size 1).

The code is slow, especially if the training sample is large, this is because we compare every "fake image" x "real image" pair. Coding optimizations are welcome.

**Needed**

* Python 3.6
* opencv-python
* scikit-image

**To run**
```bash
$ # Replace folders with yours and replace "png" by "jpg" if necessary
$ python most_similar.py --input_folder_fake_img "folder1" --input_folder_real_img "folder2" --output_folder "folder3" --fake_img_type "png" --real_img_type' "png"
```

**Example**

I chose 10 cute looking 128x128 images of cats using a trained DCGAN with SELU and looked for the 5 most similar cats. Here's the results (Top row contains the generated (fake) cats):
![](/images/DCGAN_SELU_128X128_most_similar_images1.png)
![](/images/DCGAN_SELU_128X128_most_similar_images2.png)
