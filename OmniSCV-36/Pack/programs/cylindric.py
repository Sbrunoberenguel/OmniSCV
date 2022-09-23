import cv2
import numpy as np
import programs.functions as f
from tqdm import tqdm
from programs.utils import *
import programs.fichero as fichero

#Main program
def main(scene,common=[1024,512,0,1,'cam_rot.txt','R',20.0], specific=[180,120]):

	#----------------------------------------------------------------------------
	#Cylindric image parameters
	final_w = common[0]	#Image resolution: width
	final_h = common[1]	#Image resolution: height
	init_loc = common[2]	#First location to evaluate
	num_locs = common[3]	#Number of locations
	loc_list = [i + init_loc for i in range(num_locs)] 	#List of locations
	rot1 = common[4]
	rot2 = common[5]
	thresh = float(common[6])

	hor_FOV = np.deg2rad(specific[0])	#horizontal field of view
	ver_FOV = np.deg2rad(specific[1])	#vertical Field of view
	horizontal,vertical = specific[0],specific[1]	
	#-----------------------------------------------------------------------------------------------
	if not cv2.useOptimized():
		print('Turn on the Optimizer')
		cv2.setUseOptimized(True)
	
	#Geometric parameters
	Nor,Rot = f.load_geom()
	locations = open('cam_loc.txt','r').read().split('\n')

	#Camera images
	for loc in loc_list:
		final = central_image(final_w,final_h)
		final.scene = scene
		final.location = locations[loc].split(' ')
		final.cam_type = 'cylindric'

		f.folders('cylindric','class',loc)
		
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
		R_cam = np.array([[0,-1,0],[0,0,-1],[1,0,0]]) 
		final.rotation = f.camera_rotation(rot1,rot2,loc) #Rotation matrix of the viewer
		R_world = np.dot(final.rotation,R_cam)
		FOV = np.pi/2.0
		K_RGB = np.array([[(im_w_RGB/2.0)/np.tan(FOV/2.0),0,im_w_RGB/2.0],
						  [0,(im_h_RGB/2.0)/np.tan(FOV/2.0),im_h_RGB/2.0],
						  [0,0,1]])
		K_S = np.array([[(im_w_S/2.0)/np.tan(FOV/2.0),0,im_w_S/2.0],
						[0,(im_h_S/2.0)/np.tan(FOV/2.0),im_h_S/2.0],
						[0,0,1]])
		final.max_depth = float(thresh)

		print('Composing cylindric image {} of {}'.format(loc,loc_list[-1]))
		#Pixel mapping
		x,y = np.meshgrid(np.arange(final_w),np.arange(final_h))
		theta = (0.5-x/float(final_w))*hor_FOV
		phi = ver_FOV*(0.5 - y/float(final_h))
		ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
		vec = np.array([(cp*ct),(cp*st),(sp)]).reshape(3,final_w*final_h)
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
		image_path_RGB = "cylindric/lit/"
		image_path_S = "cylindric/object_mask/"
		image_path_D = "cylindric/depth_data/"
		image_path_Dc = "cylindric/depth/"
		image_path_pc = "cylindric/point_cloud/"
		model_path = "cylindric/class/"
		image_name = "{}-cyl-{}-v{}-h{}-{}".format(scene,rot1[0]+rot2,vertical,horizontal,loc)
		#final.save_RGB(image_path_RGB,image_name)
		#final.save_S(image_path_S,image_name)
		#final.save_D(image_path_D,image_name)
		#final.inv_depth_map(image_path_Dc,image_name)
		#final.coded_depth(image_path_Dc,image_name)
		#final.save_PC(image_path_pc,image_name)
		final.save_model(model_path,image_name)
		fichero.main(final,model_path,image_name,specific)		

