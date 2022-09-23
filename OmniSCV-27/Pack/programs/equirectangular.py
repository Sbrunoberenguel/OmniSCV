import cv2
import numpy as np
import functions as f
import os
import fichero

#Auxiliar functions of equirectangular image
#Main program
def main(scene,common=[1024,512,['lit'],0,1,'cam_rot.txt','R','No','No','No'],specific=[]):

	#----------------------------------------------------------------------------
	#Equirectangular image parameters
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
	#----------------------------------------------------------------------------
	if not cv2.useOptimized():
		print('Turn on the Optimizer')
		cv2.setUseOptimized(True)
	#Geometric parameters
	Nor,Rot = f.load_geom()

	#Camera images - skybox
	for mode in mode_list:
		print('{} mode composition'.format(mode.capitalize()))
		for loc in loc_list:
			final = np.zeros((final_h,final_w,3), np.uint8)
			r,g,b = np.zeros(final_h*final_w),np.zeros(final_h*final_w),np.zeros(final_h*final_w)
			if data == "Yes" and mode=='depth':
				f.folders('equirectangular','depth_data',loc)
				depth = np.zeros((final_w*final_h,1),np.float32)
				point_cloud = np.zeros((final_w*final_h,3),np.float32)
				depth_pool = f.depth_file(loc)
			f.folders('equirectangular',mode,loc)
			imagenes = f.load_img(mode,loc)
			im_h,im_w,ch = imagenes[0].shape
			im_w -=1
			im_h -=1

			#Camera parameters
			R_cam =np.array([[0,-1,0],[0,0,-1],[1,0,0]])
			R_view = f.camera_rotation(rot1,rot2,loc) #Rotation matrix of the viewer
			R_world = np.dot(R_view,R_cam)
			FOV = np.pi/2.0
			fx = (im_w/2.0)/np.tan(FOV/2.0)
			fy = (im_h/2.0)/np.tan(FOV/2.0)
			K = np.array([[fx,0,im_w/2.0],[0,fy,im_h/2.0],[0,0,1]])
			
			print('Composing equirectangular image {} of {}'.format(loc,loc_list[-1]))
			#Pixel mapping
			x,y = np.meshgrid(np.arange(final_w),np.arange(final_h))
			theta = (1.0-2*x/float(final_w))*np.pi
			phi = (0.5-y/float(final_h))*np.pi
			ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
			vec = np.array([(cp*ct),(cp*st),(sp)]).reshape(3,final_w*final_h)
			v_abs = np.dot(R_world,vec)
			img_index = f.get_index(v_abs)
			for i in range(img_index.size):
				n,imagen,R = Nor[img_index[i]],imagenes[img_index[i]],Rot[img_index[i]]
				p_x, p_y = f.get_pixel(v_abs[:,i], R, K)
				color = imagen[p_y, p_x]
				r[i],g[i],b[i] = color[0:3]
				if data == 'Yes' and mode=='depth':
					depth[i] = depth_pool[img_index[i]][p_y,p_x]
			final = cv2.merge((r,g,b)).reshape(final_h,final_w,3)
			#Save image
			image_path = "equirectangular/{}/".format(mode)
			image_name = "{}-equirec-{}-{}".format(scene,rot1[0]+rot2,loc)
			if data == "Yes" and mode=='depth':
				v = np.transpose(v_abs)
				point_cloud=np.multiply(v,depth)
				depth = depth.reshape(final_h,final_w)
				np.save("equirectangular/depth_data/{}.npy".format(image_name),depth)
				np.save("equirectangular/depth_data/{}-pcloud-{}-{}.npy".format(scene,rot1[0]+rot2,loc),point_cloud)
			if show == 'Yes':
				cv2.imshow('{}'.format(image_name),final)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
			cv2.imwrite('{}.png'.format(image_path+image_name),final)
			fichero.main(image_path+image_name,'equirectangular',common,specific,R_view,mode,loc)		
			if semantica=='Yes' and mode == "object_mask":
				result_path = "equirectangular/binary_masks/"
				if not os.path.exists("{}".format(result_path+image_name)):
					os.makedirs("{}".format(result_path+image_name))
				f.build_binary_mask(scene,image_path+image_name+".png","{}{}/".format(result_path,image_name))
