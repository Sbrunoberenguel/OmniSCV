import cv2
import numpy as np
import math
import functions as f
import os
import fichero

#Main program
def main(scene,common=[1024,1024,['lit'],0,1,'cam_rot.txt','R','No','No','No'], specific=[-2.736690e+02,0.000000e+00,1.491215e-03,-1.810410e-06,6.614668e-09]):
	#----------------------------------------------------------------------------
	#Scaramuzza image parameters
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
	
	a0,a1,a2,a3,a4 = specific
	#----------------------------------------------------------------------------

	#Geometric parameters
	Nor,Rot = f.load_geom()

	#Camera images - skybox
	for mode in mode_list:
		for loc in loc_list:
			final = np.zeros((final_h,final_w,3), np.uint8)
			r,g,b = np.zeros(final_h*final_w),np.zeros(final_h*final_w),np.zeros(final_h*final_w)
			if data == "Yes" and mode=='depth':
				f.folders('scaramuzza','depth_data',loc)
				depth = np.zeros((final_w*final_h,1),np.float32)
				point_cloud = np.zeros((final_w*final_h,3),np.float32)
				depth_pool = f.depth_file(loc)
			f.folders('scaramuzza',mode,loc)
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
			fx = (im_w/2.0)/np.tan(FOV/2.0)
			fy = (im_h/2.0)/np.tan(FOV/2.0)
			K = np.array([[fx,0,im_w/2.0],[0,fy,im_h/2.0],[0,0,1]])
			print('Composing Scaramuzza image {} of {}'.format(loc,loc_list[-1]))
			#Pixel mapping
			x_0,y_0 = final_w/2.0,final_h/2.0
			x,y = np.meshgrid(np.arange(final_w),np.arange(final_h))
			u,v = x-x_0,y_0-y
			rho = np.sqrt(np.square(u)+np.square(v))
			out = rho>r_max
			out = out.reshape(1,rho.size)
			f_rho = a0 + a1*rho + a2*rho**2 + a3*rho**3 + a4*rho**4
			norm = np.sqrt(np.square(u)+np.square(v)+np.square(f_rho))
			vec = np.array([u/norm,v/norm,f_rho/norm]).reshape(3,final_w*final_h)
			v_abs = np.dot(R_world,vec)
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
			image_path = "scaramuzza/{}/".format(mode)
			image_name = "{}-scaramuzza-{}-{}-{}".format(scene,mode,rot1[0]+rot2,loc)
			if data == 'Yes' and mode=='depth':
				v=np.traspose(v_abs)
				point_cloud=np.multiply(v,depth)
				depth = depth.reshape(final_h,final_w)
				np.save("scaramuzza/depth_data/{}-scaramuzza-data-{}-{}.npy".format(scene,rot1[0]+rot2,loc),depth)
				np.save("scaramuzza/depth_data/{}-pcloud-{}-{}-{}.npy".format(scene,mode,rot1[0]+rot2,loc),point_cloud)
			if show == 'Yes':
				cv2.imshow('{}'.format(image_name),final)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
			cv2.imwrite('{}.png'.format(image_path+image_name),final)
			fichero.main(image_path+image_name,'scaramuzza',common,specific,R_view,mode,loc) 
			if semantica=='Yes':
				result_path = "scaramuzza/binary_masks/"
				if not os.path.exists("{}".format(result_path+image_name)):
					os.makedirs("{}".format(result_path+image_name))
				f.build_binary_mask(scene,image_path+image_name+".png","{}{}/".format(result_path,image_name))

