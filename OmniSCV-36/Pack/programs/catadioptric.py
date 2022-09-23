import cv2
import numpy as np
import programs.functions as f
from tqdm import tqdm
from programs.utils import *
import programs.fichero as fichero

#Auxiliar functions for catadioptric image composition
def h_inverse(vec,e):
	x_v = np.array(vec[0,:])
	y_v = np.array(vec[1,:])
	z_v = np.array(vec[2,:])
	norm = np.square(x_v) + np.square(y_v) + np.square(z_v)
	sqrt = np.sqrt(np.square(z_v) + (1-np.square(e))*(np.square(x_v) + np.square(y_v)))
	x = ((z_v*e + sqrt)*x_v)/norm
	y = ((z_v*e + sqrt)*y_v)/norm
	z = ((z_v*e + sqrt)*z_v)/norm - e
	vector = np.array([[x],[y],[z]]).reshape(3,vec.shape[1])
	return vector

#Main program
def main(scene,common=[1024,1024,0,1,'cam_rot.txt','R',20.0], specific=['hyper',0.1,0.1]):

	#----------------------------------------------------------------------------
	#Catadioptric image parameters
	final_w = common[0]	#Image resolution: width
	final_h = common[1]	#Image resolution: height
	init_loc = common[2]	#First location to evaluate
	num_locs = common[3]	#Number of locations
	loc_list = [i + init_loc for i in range(num_locs)] 	#List of locations
	rot1 = common[4]
	rot2 = common[5]
	thresh = float(common[6])
	
	system = specific[0]
	d = specific[1]
	lr = specific[2]
	#----------------------------------------------------------------------------
	if not cv2.useOptimized():
		print('Turn on the Optimizer')
		cv2.setUseOptimized(True)

	#Catadioptric system parameters
	p = lr/4.0
	if system == 'para':
		xi = 1
		nu = -20*p
	elif system =='hyper':
		xi = d/(np.sqrt(np.square(d)+4*np.square(p)))
		nu = -2*p/(np.sqrt(np.square(d)+4*np.square(p))) 
	elif system == 'planar':
		xi = 0
		nu = 1

	#Geometric parameters
	Nor,Rot = f.load_geom()
	locations = open('cam_loc.txt','r').read().split('\n')

	#Camera images - skybox
	for loc in loc_list:
		final = central_image(final_w,final_h)
		final.scene = scene
		final.location = locations[loc].split(' ')
		final.cam_type = 'catadioptric_'+system

		f.folders('catadioptric','class',loc)
		
		final.Colour_pool = f.load_img('lit',loc)
		final.Semantic_pool = f.load_img('object_mask',loc)
		final.Depth_pool = f.depth_file(loc)
		im_h_RGB,im_w_RGB,ch_RGB = final.Colour_pool[0].shape
		im_h_S,im_w_S,ch_S = final.Semantic_pool[0].shape
		im_h_RGB -=1
		im_w_RGB -=1
		im_h_S -=1
		im_w_S -=1

#Camera parameters
		R_cam = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
		final.rotation = f.camera_rotation(rot1,rot2,loc) #Rotation matrix of the viewer
		R_world = np.dot(final.rotation,R_cam)
		FOV = np.pi/2.0
		r_max = final_w / 2.0
		K_RGB = np.array([[(im_w_RGB/2.0)/np.tan(FOV/2.0),0,im_w_RGB/2.0],
						  [0,(im_h_RGB/2.0)/np.tan(FOV/2.0),im_h_RGB/2.0],
						  [0,0,1]])
		K_S = np.array([[(im_w_S/2.0)/np.tan(FOV/2.0),0,im_w_S/2.0],
						[0,(im_h_S/2.0)/np.tan(FOV/2.0),im_h_S/2.0],
						[0,0,1]])
		final.max_depth = float(thresh)
		fcx,fcy = (final_w/2.0)/np.tan(FOV/2.0),(final_h/2.0)/np.tan(FOV/2.0)
		Kc = np.array([[fcx,0,final_w/2.0],[0,fcy,final_h/2.0],[0,0,1]])
		Mc = np.array([[-nu,0,0],[0,nu,0],[0,0,1]])
		Hc = np.dot(Kc,Mc)
		Hc_inv = np.linalg.inv(Hc)
		print('Composing catadioptric image {} of {}'.format(loc,loc_list[-1]))
#Pixel mapping
		x_0,y_0 = final_w/2.0,final_h/2.0
		x,y = np.meshgrid(np.arange(final_w),np.arange(final_h))
		z = np.full_like(x,1)
		r_hat = np.sqrt(np.square(x-x_0)+np.square(y-y_0))
		out = r_hat>r_max
		out = out.reshape(1,r_hat.size)
		vec_hat = np.array([[x],[y],[z]]).reshape(3,final_w*final_h)
		vec_norm = np.dot(Hc_inv,vec_hat)
		vec = h_inverse(vec_norm,xi)
		final.vec = np.dot(R_world,vec)
		img_index = f.get_index(final.vec)
		rot_rot = np.array([Rot[i] for i in img_index])
		rot_vec = np.transpose(np.array([np.dot(rot_rot[i],final.vec[:,i]) for i in range(rot_rot.shape[0])]))		
		pixels_RGB = np.dot(K_RGB,rot_vec)
		pixels_S = np.dot(K_S,rot_vec)	
		p_x_RGB, p_y_RGB = pixels_RGB[0,:]/pixels_RGB[2,:], pixels_RGB[1,:]/pixels_RGB[2,:]
		p_x_S, p_y_S = pixels_S[0,:]/pixels_S[2,:], pixels_S[1,:]/pixels_S[2,:]
		
		for i in tqdm(range(rot_vec.shape[-1])):
			final.RGB[:,i] = final.Colour_pool[img_index[i]][int(p_y_RGB[i]),int(p_x_RGB[i])]
			final.S[:,i] = final.Semantic_pool[img_index[i]][int(p_y_S[i]),int(p_x_S[i])]
			final.D_data[:,i] = final.Depth_pool[img_index[i]][int(p_y_RGB[i]),int(p_x_RGB[i])]
		
		#Save image
		image_path_RGB = "catadioptric/lit/"
		image_path_S = "catadioptric/object_mask/"
		image_path_D = "catadioptric/depth_data/"
		image_path_Dc = "catadioptric/depth/"
		image_path_pc = "catadioptric/point_cloud/"
		model_path = "catadioptric/class/"
		image_name = "{}-{}cata-{}-d{}-2p{}-{}".format(scene,system,rot1[0]+rot2,int(d*100),int(2*p*100),loc)
		#Final save
		#final.save_RGB(image_path_RGB,image_name)
		#final.save_S(image_path_S,image_name)
		#final.save_D(image_path_D,image_name)
		#final.inv_depth_map(image_path_Dc,image_name)
		#final.coded_depth(image_path_Dc,image_name)
		#final.save_PC(image_path_pc,image_name)
		final.save_model(model_path,image_name)
		fichero.main(final,model_path,image_name,specific)
			

