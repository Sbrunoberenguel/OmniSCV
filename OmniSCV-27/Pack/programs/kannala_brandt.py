import cv2
import numpy as np
import functions as f
import os
from scipy.optimize import curve_fit
import fichero

#Functions

def full_func(x,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9):
	return a0+a1*x+a2*x**2+a3*x**3+a4*x**4+a5*x**5+a6*x**6+x*a7**7+a8*x**8+x*a9**9

def half_func(x,a0,a1,a2,a3,a4,a5):
	return a0+a1*x+a2*x**2+a3*x**3+a4*x**4+a5*x**5

#Computes the focal length for the different fish_eye systems
def focal_lenght(system,r_max,phi):
	if system == 'equiang':
		f = r_max/float(phi)
	elif system == 'stereo':
		f = r_max/float(2*np.tan(phi/2.0))
	elif system == 'orth':
		f = r_max/float(np.sin(phi))
	elif system == 'equisol':
		f = r_max/float(np.sin(phi/2.0))
	else:
		print('Camera system ERROR')
		f = 0
	return f

#Computes 10 points from a fish_eye system
def points_10(system,r_max,phi_max):
	x,y = [],[]
	f = focal_lenght(system,r_max,phi_max)
	for r in np.arange(0,r_max,r_max/10.0):
		aux = r/f
		if system == 'equiang':
			func = aux
		elif system == 'stereo':
			func = 2*np.arctan2(r,2*f)
		elif system == 'orth':
			if aux>1:
				r *= 10
				aux = 1
			func = np.arcsin(aux)
		elif system == 'equisol':
			if aux>1:
				r *= 10
				aux = 1
			func = 2*np.arcsin(aux)
		else:
			print('Camera system ERROR')
		x.append(r)
		y.append(func)
	return x,y

#Get the kannala-brandt parameters of the model
def kannala_brandt(system,r_max,phi_max,func):
	model = []
	x,y = points_10(system,r_max,phi_max)
	if func == 'full':
		kb,pconv = curve_fit(full_func, x, y, bounds=([0,0.9999999,0,-np.inf, 0, -np.inf,0,-np.inf,0,-np.inf], [0.0000001,1,0.0000001,np.inf, 0.0000001, np.inf,0.0000001,np.inf,0.0000001,np.inf]))
	else:
		kb,pconv = curve_fit(half_func, x, y, bounds=([0,0.9999999,0,-np.inf, 0, -np.inf], [0.0000001,1,0.0000001,np.inf, 0.0000001, np.inf]))
	for i in np.arange(1,len(kb)):
		model.append(kb[-i])
	return model

def newton(model,r_u,init,func):
	prec = np.full_like(r_u,0.0001)
	x_n = init
	for i in range(20):
		x = x_n
		if func == 'full':
			fun = model[0]*x**9 + model[2]*x**7 + model[4]*x**5 + model[6]*x**3 + model[8]*x + r_u
			dif = 9*model[0]*x**8 + 7*model[2]*x**6 + 5*model[4]*x**4 + 3*model[6]*x**2 + model[8]*np.full_like(r_u,1)
		else:
			fun = model[0]*x**5 + model[2]*x**3 + model[4]*x + r_u
			dif = 5*model[0]*x**4 + 3*model[2]*x**2 + model[4]*np.full_like(r_u,1)
		x_n = x - fun/dif
		err = np.absolute(x_n-x)
		if err.all() < prec.all():
			return x_n
	return x_n
			
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
	#Kannala-Brandt model for fish_eye systems
	func = 'half'
	model = kannala_brandt(system,final_w/2.0,FOV_fish/2.0,func)

	if not cv2.useOptimized():
		print('Turn on the Optimizer')
		cv2.setUseOptimized(True)

	#Geometric parameters
	Nor,Rot = f.load_geom()

	#Camera images - cubemap
	for mode in mode_list:
		print('{} mode composition'.format(mode.capitalize()))
		for loc in loc_list:
			if data == "Yes" and mode=='depth':
				f.folders('kannala','depth_data',loc)
				depth = np.zeros((final_w*final_h,1),np.float32)
				point_cloud = np.zeros((final_w*final_h,3),np.float32)
				depth_pool = f.depth_file(loc)
			final = np.zeros((final_h,final_w,3), np.uint8)
			r,g,b = np.zeros(final_h*final_w),np.zeros(final_h*final_w),np.zeros(final_h*final_w)
			f.folders('kannala',mode,loc)
			imagenes = f.load_img(mode,loc)
			im_h,im_w,ch = imagenes[0].shape
			im_w -=1
			im_h -=1

	#Camera parameters
			R_cam = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
			R_view = f.camera_rotation(rot1,rot2,loc) #Rotation matrix of the viewer
			R_world = np.dot(R_view,R_cam)
			r_max = final_w / 2.0
			f_fish = focal_lenght(system,r_max,FOV_fish/2.0)
			FOV = np.pi/2.0
			fx = (im_w/2.0)/np.tan(FOV/2.0)
			fy = (im_h/2.0)/np.tan(FOV/2.0)
			K = np.array([[fx,0,im_w/2.0],[0,fy,im_h/2.0],[0,0,1]]) 
			print('Composing Kannala-Brandt image {} of {}'.format(loc,loc_list[-1]))
	#Pixel mapping
			x_0,y_0 = final_w/2.0,final_h/2.0
			x,y = np.meshgrid(np.arange(final_w),np.arange(final_h))
			mx = (x-x_0)/f_fish
			my = (y-y_0)/f_fish
			theta = np.arctan2(my,mx)
			r_u = np.sqrt(np.square(mx)+np.square(my))
			init = np.full_like(r_u,FOV_fish/2.0)
			phi = newton(model,r_u,init,func)
			r_hat = np.sqrt(np.square(x_0-x)+np.square(y-y_0))
			out = r_hat>r_max
			out = out.reshape(1,r_hat.size)
			ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
			vec = np.array([(sp*ct),(sp*st),(cp)]).reshape(3,final_w*final_h)
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
			image_path = "kannala/{}/".format(mode)
			image_name = "{}-kannala_brandt_{}-{}-FOV{}-{}".format(scene,system,rot1[0]+rot2,specific[1],loc)
			if data == "Yes" and mode=='depth':
				v = np.transpose(v_abs)
				point_cloud=np.multiply(v,depth)
				depth = depth.reshape(final_h,final_w)
				np.save("kannala/depth_data/{}.npy".format(image_name),depth)
				np.save("kannala/depth_data/{}-pcloud-{}-{}-FOV{}-{}".format(scene,system,rot1[0]+rot2,specific[1],loc),point_cloud)
			if show == 'Yes':
				cv2.imshow('{}'.format(image_name),final)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
			cv2.imwrite('{}.png'.format(image_path+image_name),final)
			fichero.main(image_path+image_name,'kannala '+system,common,specific,R_view,mode,loc) 
			if semantica=='Yes' and mode == "object_mask":
				result_path = "kannala/binary_masks/"
				if not os.path.exists("{}".format(result_path+image_name)):
					os.makedirs("{}".format(result_path+image_name))
				f.build_binary_mask(scene,image_path+image_name+".png","{}{}/".format(result_path,image_name))
