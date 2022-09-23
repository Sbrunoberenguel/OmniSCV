import cv2
import open3d as o3d
import numpy as np
from scipy.stats import special_ortho_group as sop
import os
from tqdm import trange

def model_3D(point_cloud):
	name = point_cloud.split('/')[-1]
	rgb_file = point_cloud.split('/')[0]+'/RGB/'+name[:-3]
	rgb = cv2.imread(rgb_file+'png')
	RGB = rgb.reshape(-1,3)/255.0
	pc = np.load(point_cloud)
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pc)
	pcd.colors = o3d.utility.Vector3dVector(RGB)
	o3d.io.write_point_cloud('{}.ply'.format(point_cloud[0:-4]),pcd)
	o3d.visualization.draw_geometries([pcd])
	
#Creates binary mask for each object
def build_binary_mask(scene,image_name):
	img_path = image_name.split('/')[0]
	img_name = scene+image_name.split(scene)[1]
	result_path = os.path.join(img_path,'binary_mask',img_name[:-4])
	if not os.path.exists(result_path):
		os.makedirs(result_path)
	f = open('mask_list_{}.txt'.format(scene),'r')
	obj_file = f.read().split('\n')
	image = cv2.imread(image_name)
	im_h,im_w,_ = image.shape
	th_min = 1e3
	th_max = 1e10
	gap = 0
	img = image.reshape(-1,3)
	clist,freq = np.unique(img,axis=0,return_counts=True)
	for item in trange(len(obj_file)-1):
		candidate = []
		mask = np.zeros((im_h*im_w))
		obj = obj_file[item].split(' ')
		obj_col = np.array([int(obj[1]),int(obj[2]),int(obj[3])])
		search = np.isclose(obj_col,clist,atol=gap)
		for i in range(search.shape[0]):
			if search[i].all():
				candidate.append(i)
		if len(candidate)==0:
			continue
		elif len(candidate)==1:
			idx = candidate[0]
		else:
			aux = np.where(freq[candidate]==np.amax(freq[candidate]))[0]
			idx = candidate[int(aux)]
		if th_min < freq[idx] < th_max:
			out = np.isclose(clist[idx],img)
			for j in range(out.shape[0]):
				if out[j].all():
					mask[j] = 255
			cv2.imwrite(os.path.join(result_path,obj[0]+'.png'),mask.reshape(im_h,im_w))
		else:
			continue


def video(scene):
	#Video Set-up
	pano = input('Set kind of omnidirectional image for the video: ').lower()
	frames = input('Number of frames: ')
	mode = input('Set image mode lit-object_mask: ')
	name = mode[0:3]
	view = input('What is the view direction? (ej: xpos): ') 
	SO = input('Operative system?: ').capitalize()

#Video building
	pano_video = []
	for i in range(frames):
		pano_img = cv2.imread('{}/{}/{}-{}-{}-{}.png'.format(pano,mode,scene,pano,view,i+1))
		pano_video.append(pano_img)

	hp,wp,_ = pano_video[0].shape

	if SO == 'Linux':
		out_pano = cv2.VideoWriter('{}_{}_video.avi'.format(pano,name),cv2.VideoWriter_fourcc(*"MJPG"),30.0,(wp,hp))
	else:
		out_pano = cv2.VideoWriter('{}_{}_video.mp4'.format(pano,name),cv2.VideoWriter_fourcc(*"mp4v"),30.0,(wp,hp))
		
	for i in range(frames):
		out_pano.write(pano_video[i])

	out_pano.release()

def randomRotations():
	rot = open('cam_rot.txt','w')
	loc = open('cam_loc.txt','r').read().split('\n')
	for i in range(len(loc)):
		R = sop.rvs(3)
		rot.write('RL:{} {} {} {} {} {} {} {} {}\n'.format( 
								R[0,0],R[0,1],R[0,2],
								R[1,0],R[1,1],R[1,2],
								R[2,0],R[2,1],R[2,2]))
	rot.close()


