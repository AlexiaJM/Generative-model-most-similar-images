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
from os import cpu_count
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder_fake_img', default='/home/alexia/fake_cats', help='Folder with your fake images (generated with batch_size = 1)')
parser.add_argument('--input_folder_real_img', default='/home/alexia/Datasets/Meow_64x64/cats_bigger_than_64x64', help='Folder with the real images (With cats, make sure it is the folder of the cropped cat faces, not the original cats)')
parser.add_argument('--output_folder', default='/home/alexia/most_similar_cats', help='Output folder')
parser.add_argument('--fake_img_type', default='jpg', help='File extension of fake images')
parser.add_argument('--real_img_type', default='jpg', help='File extension of real images')
parser.add_argument('--out_img_type', default='png', help='File extension of output')
parser.add_argument('--cpus', type=int, default=-1, help='CPUs number')
parser.add_argument('--img_size', type=int, default=0, help='If equal to 0, will use the size of the smallest fake image. Otherwise, will force a specific size.')
param = parser.parse_args()

if param.cpus == -1:
	param.cpus = cpu_count() - 1

## Imports

import cv2
import os
import glob
import math
import sys
import operator
import numpy as np
from skimage.metrics import structural_similarity
#from skimage.measure import compare_ssim # Only for old versions

from multiprocessing import Manager
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial


## UTILITY FUNCTIONS

# Split the list in chunks
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Compare a single fake image with a list of original images, each process will return a local 
def compare_images(original_images, image_fake):
	most_similar_images = ["", "", "", "", ""]
	most_similar_images_ssim = [-1, -1, -1, -1, -1]
	for imagePath_real in original_images:
		# Get real image
		image_real = cv2.imread(imagePath_real)
		# Resize to same as the fake
		image_real = cv2.resize(image_real, (img_size_h, img_size_w))
		# compare fake from real and store if more similar than the least similar of top 5
		similarity = structural_similarity(image_fake, image_real, multichannel=True)
		min_index, min_similarity = min(enumerate(most_similar_images_ssim), key=operator.itemgetter(1))
		if similarity > min_similarity:
			most_similar_images[min_index] = imagePath_real
			most_similar_images_ssim[min_index] = similarity
	for index,image_name in enumerate(most_similar_images):
		most_similar_images_shared[image_name] = most_similar_images_ssim[index]



## BODY

# All input images must be the same size
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
mgr = Manager()

# Store all real images paths
real_images_paths = glob.glob('%s/*.%s' % (param.input_folder_real_img,param.real_img_type))
real_images_paths = list(chunks(real_images_paths, int(len(real_images_paths)/param.cpus)))

for imagePath_fake in glob.glob('%s/*.%s' % (param.input_folder_fake_img,param.fake_img_type)):

	# Init shared dictionary to store images similarity
	most_similar_images_shared = mgr.dict()

	# Get fake image and size
	image_fake = cv2.imread(imagePath_fake)
	image_fake = cv2.resize(image_fake, (img_size_h, img_size_w))
	
	# Compare fake image with all the real images using multiprocessing
	with Pool(processes=param.cpus) as p:
		with tqdm(total=len(real_images_paths)) as pbar:
			for v in p.imap_unordered(partial(compare_images, image_fake = image_fake), real_images_paths):
				pbar.update()

	# Sort from most similar to least similar
	most_similar_images = {k: v for k, v in sorted(most_similar_images_shared.items(), key=lambda item: item[1])}

	# Extract the most similar 5 images
	most_similar_images = list(most_similar_images)[-5:]
	most_similar_images.reverse()

	# Add most similar ones to picture
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
