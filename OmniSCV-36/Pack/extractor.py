from programs.utils import *
import numpy as np
import cv2
import open3d as o3d
import argparse
import os, pickle

def main(img, args):
	image_model = pickle.load(open(args.in_dir+'/'+img,'rb'))
	image_name = img.split('.')[0]
	if args.rgb:
		if not os.path.isdir(args.out_dir+'/RGB/'):
			os.makedirs(args.out_dir+'/RGB/')
		image_model.save_RGB(args.out_dir+'/RGB/',image_name)
	if args.s:
		if not os.path.isdir(args.out_dir+'/object_mask/'):
			os.makedirs(args.out_dir+'/object_mask/')		
		image_model.save_S(args.out_dir+'/object_mask/',image_name)
	if args.d:
		if not os.path.isdir(args.out_dir+'/depth_data/'):
			os.makedirs(args.out_dir+'/depth_data/')
		image_model.save_D(args.out_dir+'/depth_data/',image_name)
	if args.inv:
		if not os.path.isdir(args.out_dir+'/depth/'):
			os.makedirs(args.out_dir+'/depth/')		
		image_model.inv_depth_map(args.out_dir+'/depth/',image_name)
	if args.cod:
		if not os.path.isdir(args.out_dir+'/depth/'):
			os.makedirs(args.out_dir+'/depth/')		
		image_model.coded_depth(args.out_dir+'/depth/',image_name)
	if args.pc:
		if not os.path.isdir(args.out_dir+'/point_cloud/'):
			os.makedirs(args.out_dir+'/point_cloud/')		
		image_model.save_PC(args.out_dir+'/point_cloud/',image_name)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--in_dir', required=True, help='In class directory')
	parser.add_argument('--out_dir', required=True, help='Output directory')
	parser.add_argument('-rgb','--rgb', help='Extracts RGB image', action='store_true', default=False)
	parser.add_argument('-s','--s', help='Extracts Semantic image', action='store_true', default=False)
	parser.add_argument('-d','--d', help='Extracts depth data', action='store_true', default=False)
	parser.add_argument('-inv','--inv', help='Extracts inverse depth map', action='store_true', default=False)
	parser.add_argument('-cod', '--cod', help='Extracts coded depth image', action='store_true', default=False)
	parser.add_argument('-pc', '--pc', help='Extracts Point Cloud', action='store_true', default=False)
	
	args = parser.parse_args()

	if not os.path.isdir(args.in_dir):
		print('This path does not exist')

	if not os.path.isdir(args.out_dir):
		os.makedirs(args.out_dir)

	images = os.listdir(args.in_dir)
	classes = []
	for img in images:
		if '.pkl' in img:
			classes.append(img)
	for img in classes:
		main(img,args)


