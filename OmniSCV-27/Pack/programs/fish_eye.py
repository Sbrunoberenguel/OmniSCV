import cv2
import numpy as np
import functions as f
import os
import fichero

#Auxiliar functions for fish eye image composition
#Computes the parameter PHI of the fish-eye
def cam_system(system,r,f):
	aux = r/f
	if system == 'equiang':
		return aux
	elif system == 'stereo':
		return 2*np.arctan2(r,2*f)
	elif system == 'orth':
		return np.arcsin(aux)
	elif system == 'equisol':
		return 2*np.arcsin(aux)
	else:
		print('Camera system ERROR')

#Computes focal length for each fish eye sytem
def focal_lenght(system,r_max,phi):
	if system == 'equiang':
		f = r_max/float(phi)
	elif system == 'stereo':
		f = r_max/float((2*np.tan(phi/2.0)))
	elif system == 'orth':
		f = r_max/float(np.sin(phi))
	elif system == 'equisol':
		f = r_max/float(np.sin(phi/2.0))
	else:
		print('Camera system ERROR')
		f = 0
	return f


#Main program
def main(scene,common=[1024,1024,['lit'],0,1,'cam_rot.txt','R','No','No','No'],specific=['equiang',220]):
	
	#----------------------------------------------------------------------------
	#fishe eye image parameters
	final_w = common[0]	#Image resolution: width
	final_h = common[1]	#Image resolution: height
	mode_list = common[2] #View mode
	init_loc = common[3]	#First location to evaluate
	num_locs = common[4]	#Number of locations
	loc_list = [i + init_loc for i in range(num_locs)] 	#List of locations
	rot1 = common[5]
	rot2 = common[6]
	show = common[7]
	semantica = common[8]
	data = common[9]
	
	system = specific[0]
	FOV_fish = np.deg2rad(specific[1])
	#----------------------------------------------------------------------------
	if not cv2.useOptimized():
		print('Turn on the Optimizer')
		cv2.setUseOptimized(True)
	#Geometric parameters
	Nor,Rot = f.load_geom()

	#Camera images - cubemap
	for mode in mode_list:
		print('{} mode composition'.format(mode.capitalize()))
		for loc in loc_list:
			final = np.zeros((final_h,final_w,3), np.uint8)
			r,g,b = np.zeros(final_h*final_w),np.zeros(final_h*final_w),np.zeros(final_h*final_w)
			if data == "Yes" and mode=='depth':
				f.folders('fish_eye','depth_data',loc)
				depth = np.zeros((final_w*final_h,1),np.float32)
				point_cloud = np.zeros((final_w*final_h,e),np.float32)
				depth_pool = f.depth_file(loc)
			f.folders('fish_eye',mode,loc)
			imagenes = f.load_img(mode,loc)
			im_h,im_w,ch = imagenes[0].shape
			im_w -=1
			im_h -=1

			#Camera parameters
			R_view = f.camera_rotation(rot1,rot2,loc) #Rotation matrix of the viewer
			#R_world = np.dot(R_cam,R_view)
			r_max = max(final_w/2,final_h/2)
			f_fish = focal_lenght(system,r_max,FOV_fish/2.0)
			FOV = np.pi/2.0
			fx = (im_w/2.0)/np.tan(FOV/2.0)
			fy = (im_h/2.0)/np.tan(FOV/2.0)
			K = np.array([[fx,0,im_w/2.0],[0,fy,im_h/2.0],[0,0,1]])
			
			print('Composing fish eye image {} of {}'.format(loc,loc_list[-1]))
			#Pixel mapping
			x_0,y_0 = final_w/2.0,final_h/2.0
			x,y = np.meshgrid(np.arange(final_w),np.arange(final_h))
			r_hat = np.sqrt(np.square(x_0-x)+np.square(y-y_0))
			out = r_hat>r_max
			out = out.reshape(1,r_hat.size)
			theta = np.arctan2(x-x_0,y-y_0)
			phi = cam_system(system,r_hat,f_fish)
			ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
			vec = np.array([(sp*st),(sp*ct),(cp)]).reshape(3,final_w*final_h)
			v_abs = np.dot(R_view,vec)
			img_index = f.get_index(v_abs)
			for i in range(img_index.size):
				if out[0,i]:
					continue
				n,imagen,R = Nor[img_index[i]],imagenes[img_index[i]],Rot[img_index[i]]
				p_x, p_y = f.get_pixel(v_abs[:,i], R, K)
				color = imagen[p_y, p_x]
				r[i],g[i],b[i] = color[0:3]
				if data == 'Yes' and mode=='depth':
					depth[i] = depth_pool[img_index[i]][p_y,p_x]
			final = cv2.merge((r,g,b)).reshape(final_h,final_w,3)
			image_path = "fish_eye/{}/".format(mode)
			image_name = "{}-fish_eye_{}-{}-FOV{}-{}".format(scene,system,rot1[0]+rot2,specific[1],loc)
			if data == "Yes" and mode=='depth':
				v = np.transpose(v_abs)
				point_cloud=np.multiply(v,depth)
				depth = depth.reshape(final_h,final_w)
				np.save("fish_eye/depth_data/{}.npy".format(image_name),depth)
				np.save("fish_eye/depth_data/{}-pcloud-{}-{}-FOV{}-{}.npy".format(scene,system,rot1[0]+rot2,specific[1],loc),point_cloud)
			if show == 'Yes':
				cv2.imshow('{}'.format(image_name),final)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
			cv2.imwrite('{}.png'.format(image_path+image_name),final)
			fichero.main(image_path+image_name,'fish-eye '+system,common,specific,R_view,mode,loc)
			if semantica=='Yes' and mode == "object_mask":
				result_path = "fish_eye/binary_masks/"
				if not os.path.exists("{}".format(result_path+image_name)):
					os.makedirs("{}".format(result_path+image_name))
				f.build_binary_mask(scene,image_path+image_name+".png","{}{}/".format(result_path,image_name))
