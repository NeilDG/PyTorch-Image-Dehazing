import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob


def dehaze_image(results_path, image_path):

	data_hazy = Image.open(image_path)
	data_hazy = data_hazy.resize((512, 512))
	data_hazy = (np.asarray(data_hazy)/255.0)

	data_hazy = torch.from_numpy(data_hazy).float()
	data_hazy = data_hazy.permute(2,0,1)
	data_hazy = data_hazy.cuda().unsqueeze(0)

	dehaze_net = net.dehaze_net().cuda()
	dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth'))

	clean_image = dehaze_net(data_hazy)
	#torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "results/" + image_path.split("\\")[1])
	torchvision.utils.save_image(clean_image, results_path + image_path.split("\\")[1])
	

if __name__ == '__main__':

	# img_dir = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
	# test_list = glob.glob(img_dir + "*.jpg")  # specify atmosphere intensityimg_dir = "D:/Datasets/OTS_BETA/haze/"
	# results_path = "results/O-Haze/"
	#
	# for image in test_list:
	# 	dehaze_image(results_path, image)
	# 	print(image, "done!")

	img_dir = "E:/Hazy Dataset Benchmark/I-HAZE/hazy/"
	test_list = glob.glob(img_dir + "*.jpg")  # specify atmosphere intensityimg_dir = "D:/Datasets/OTS_BETA/haze/"
	results_path = "results/I-Haze/"

	for image in test_list:
		dehaze_image(results_path, image)
		print(image, "done!")

	#
	# img_dir = "D:/Datasets/OTS_BETA/haze/"
	# test_list = glob.glob(
	# 	img_dir + "*0.95_0.2.jpg")  # specify atmosphere intensityimg_dir = "D:/Datasets/OTS_BETA/haze/"
	# results_path = "results/OTS-Beta/"
	#
	# for image in test_list:
	# 	dehaze_image(results_path, image)
	# 	print(image, "done!")

	# img_dir = "E:/Hazy Dataset Benchmark/RESIDE-Unannotated/"
	# test_list = glob.glob(img_dir + "*.jpeg")  # specify atmosphere intensityimg_dir = "D:/Datasets/OTS_BETA/haze/"
	# results_path = "results/RESIDE-3/"
	#
	# for image in test_list:
	# 	dehaze_image(results_path, image)
	# 	print(image, "done!")
