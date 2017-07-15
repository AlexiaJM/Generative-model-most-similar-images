#####
##### After generating fake images (batch size of 1) with your favorite Generator, use this to see what are the 5 most similar real (training) images.
#####
## Example with cats :
# python DCGAN.py --output_folder /home/output_folder/run-5/images/ --gen_extra_images 5 --n_epoch 1 --batch_size 1 --G_load /home/output_folder/run-5/models/G_epoch_200.pth --D_load /home/output_folder/run-5/models/D_epoch_200.pth
# python most_similar_cats.py --input_folder_fake_img /home/output_folder/run-5/images/extra --input_folder_real_img "where_I_put_my_real_cats"
## Dependencies:
# pip install opencv-python
# conda install scikit-image

## Parameters

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder_fake_img', default='/home/alexia/fake_cats', help='Folder with your fake images (generated with batch_size = 1)')
parser.add_argument('--input_folder_real_img', default='/home/alexia/Datasets/Meow_64x64/cats_bigger_than_64x64', help='Folder with the real images (With cats, make sure it is the folder of the cropped cat faces, not the original cats)')
parser.add_argument('--output_folder', default='/home/alexia/most_similar_cats', help='Output folder')
parser.add_argument('--fake_img_type', default='jpg', help='File extension of fake images')
parser.add_argument('--real_img_type', default='jpg', help='File extension of real images')
parser.add_argument('--out_img_type', default='png', help='File extension of output')
parser.add_argument('--img_size', type=int, default=0, help='If equal to 0, will use the size of the smallest fake image. Otherwise, will force a specific size.')
param = parser.parse_args()

## Imports

import cv2
import os
import glob
import math
import sys
import operator
import numpy as np
from skimage.measure import compare_ssim

## All input images must be the same size
if param.img_size == 0:
	img_size_w = 0
	img_size_h = 0
	warning = False
	for imagePath_fake in glob.glob('%s/*.%s' % (param.input_folder_fake_img,param.fake_img_type)):
		# Get fake image and size
		image_fake = cv2.imread(imagePath_fake)
		w, h, colors = image_fake.shape
		if img_size_w == 0 and img_size_h == 0:
			img_size_w = w
			img_size_h = h
		elif img_size_w != w or img_size_h != h:
			img_size_w = min(img_size_w, w)
			img_size_h = min(img_size_h, h)
			if not warning:
				print("Warning, some fake images are not of the same size, they will all have the minimum size")
			warning = True
else:
	img_size_w = param.img_size
	img_size_h = param.img_size

## Main
i = 1
for imagePath_fake in glob.glob('%s/*.%s' % (param.input_folder_fake_img,param.fake_img_type)):
	# Get fake image and size
	image_fake = cv2.imread(imagePath_fake)
	image_fake = cv2.resize(image_fake, (img_size_h, img_size_w))
	# Store 5 most-similars
	most_similar_images = ["", "", "", "", ""]
	most_similar_images_ssim = [-1, -1, -1, -1, -1]
	for imagePath_real in glob.glob('%s/*.%s' % (param.input_folder_real_img,param.real_img_type)):
		# Get real image
		image_real = cv2.imread(imagePath_real)
		# Resize to same as the fake
		image_real = cv2.resize(image_real, (img_size_h, img_size_w))
		# compare fake from real and store if more similar than the least similar of top 5
		similarity = compare_ssim(image_fake, image_real, multichannel=True)
		min_index, min_similarity = min(enumerate(most_similar_images_ssim), key=operator.itemgetter(1))
		if similarity > min_similarity:
			most_similar_images[min_index] = imagePath_real
			most_similar_images_ssim[min_index] = similarity
	# Add most similar ones to picture
	most_similar_images.sort()
	current_image = image_fake
	for imagePath_real in most_similar_images:
		image_real = cv2.imread(imagePath_real)
		image_real = cv2.resize(image_real, (img_size_h, img_size_w))
		current_image = np.concatenate((current_image, image_real), axis=0)
	# If not the first fake image to be processed, add it horizontally to previous results
	if i == 1:
		combined_image = current_image
	else:
		combined_image = np.concatenate((combined_image, current_image), axis=1)
	# For progress
	print("%d image(s) processed" % i)
	i = i + 1

## Output (Make sure no file exist from before, otherwise create new file)
run = 0
output = '%s/most_similar_images%d.%s' % (param.output_folder, run, param.out_img_type)
while os.path.exists(output):
	run += 1
	output = '%s/most_similar_images%d.%s' % (param.output_folder, run, param.out_img_type)
cv2.imwrite(output, combined_image)
