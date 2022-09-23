import cv2
import numpy as np
import functions as f
import os
import fichero

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
def main(scene,common=[1024,1024,['lit'],0,1,'cam_rot.txt','R','No','No','No'],specific=['hyper',0.1,0.1]):

	#----------------------------------------------------------------------------
	#Catadioptric image parameters
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

	#Camera images - skybox
	for mode in mode_list:
		print('{} mode composition'.format(mode.capitalize()))
		for loc in loc_list:
			final = np.zeros((final_h,final_w,3), np.uint8)
			r,g,b = np.zeros(final_h*final_w),np.zeros(final_h*final_w),np.zeros(final_h*final_w)
			if data == "Yes" and mode == 'depth':
				f.folders('catadioptric','depth_data',loc)
				depth = np.zeros((final_w*final_h,1),np.float32)
				point_cloud = np.zeros((final_w*final_h,3),np.float32)
				depth_pool = f.depth_file(loc)
			f.folders('catadioptric',mode,loc)
			imagenes = f.load_img(mode,loc)
			im_h,im_w,ch = imagenes[0].shape
			im_w -=1
			im_h -=1

	#Camera parameters
			R_cam = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
			R_view = f.camera_rotation(rot1,rot2,loc) #Rotation matrix of the viewer
			R_world = np.dot(R_view,R_cam)
			FOV = np.pi/2.0
			r_max = final_w / 2.0
			fx,fy = (im_w/2.0)/np.tan(FOV/2.0),(im_h/2.0)/np.tan(FOV/2.0)
			fcx,fcy = (final_w/2.0)/np.tan(FOV/2.0),(final_h/2.0)/np.tan(FOV/2.0)
			K = np.array([[fx,0,im_w/2.0],[0,fy,im_h/2.0],[0,0,1]])
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
			v_abs = np.dot(R_world,vec)
			img_index = f.get_index(v_abs)
			for i in range(img_index.size):
				if out[0,i]:
					continue
				n,imagen,R = Nor[img_index[i]],imagenes[img_index[i]],Rot[img_index[i]]
				p_x, p_y = f.get_pixel(v_abs[:,i], R, K)
				color = imagen[p_y,p_x]
				r[i],g[i],b[i] = color[0:3]
				if data == 'Yes' and mode == 'depth':
					depth[i] = depth_pool[img_index[i]][p_y,p_x]
			final = cv2.merge((r,g,b)).reshape(final_h,final_w,3)
			image_path = "catadioptric/{}/".format(mode)
			image_name = "{}-{}cata-{}-d{}-2p{}-{}".format(scene,system,rot1[0]+rot2,int(d*100),int(2*p*100),loc)
			if data == "Yes" and mode=='depth':
				v = np.transpose(v_abs)
				point_cloud = np.multiply(v,depth)
				depth = depth.reshape(final_h,final_w)
				np.save("catadioptric/depth_data/{}.npy".format(image_name),depth)
				np.save("catadioptric/depth_data/{}-pcloud-{}-d{}-2p{}-{}".format(scene,system,rot1[0]+rot2,int(d*100),int(2*p*100),loc),point_cloud)
			if show == 'Yes':
				cv2.imshow('{}'.format(image_name),final)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
			cv2.imwrite('{}.png'.format(image_path+image_name),final) 
			fichero.main(image_path+image_name,system+'-catadioptric',common,specific,R_view,mode,loc)
			if semantica == 'Yes' and mode == "object_mask":
				result_path = "catadioptric/binary_masks/"
				if not os.path.exists("{}".format(result_path+image_name)):
					os.makedirs("{}".format(result_path+image_name))
				f.build_binary_mask(scene,image_path+image_name+".png","{}{}/".format(result_path,image_name))
