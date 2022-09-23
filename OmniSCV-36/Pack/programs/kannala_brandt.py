import cv2
import numpy as np
import programs.functions as f
from scipy.optimize import curve_fit
from tqdm import tqdm
from programs.utils import *
import programs.fichero as fichero

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
		kb,pconv = curve_fit(full_func, x, y, 
							bounds=([0,0.9999999,0,-np.inf, 0, -np.inf,0,-np.inf,0,-np.inf], 
							[0.0000001,1,0.0000001,np.inf, 0.0000001, np.inf,0.0000001,np.inf,0.0000001,np.inf]))
	else:
		kb,pconv = curve_fit(half_func, x, y, 
							bounds=([0,0.9999999,0,-np.inf, 0, -np.inf], 
							[0.0000001,1,0.0000001,np.inf, 0.0000001, np.inf]))
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
def main(scene,common=[1024,1024,0,1,'cam_rot.txt','R',20.0], specific=['equiang',160]):
	
		#----------------------------------------------------------------------------
	#fishe eye image parameters
	final_w = common[0]	#Image resolution: width
	final_h = common[1]	#Image resolution: height
	init_loc = common[2]	#First location to evaluate
	num_locs = common[3]	#Number of locations
	loc_list = [i + init_loc for i in range(num_locs)] 	#List of locations
	rot1 = common[4]
	rot2 = common[5]
	thresh = float(common[6])

	system = specific[0]
	FOV_fish = np.deg2rad(specific[1])
	
	r_max = min(final_w/2.0,final_h/2.0)
	f_fish = focal_lenght(system,r_max,FOV_fish/2.0)
	#----------------------------------------------------------------------------
	#Kannala-Brandt model for fish_eye systems
	func = 'half'
	model = kannala_brandt(system,r_max,FOV_fish/2.0,func)
	specific.append(model)
	
	if not cv2.useOptimized():
		print('Turn on the Optimizer')
		cv2.setUseOptimized(True)

	#Geometric parameters
	Nor,Rot = f.load_geom()
	locations = open('cam_loc.txt','r').read().split('\n')

	#Camera images - cubemap
	for loc in loc_list:
		final = central_image(final_w,final_h)
		final.scene = scene
		final.location = locations[loc].split(' ')
		final.cam_type = 'kannala_'+system

		f.folders('kannala','class',loc)
		
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
		R_cam = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
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
		#out = r_hat>r_max
		#out = out.reshape(1,r_hat.size)
		ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
		vec = np.array([(sp*ct),(sp*st),(cp)]).reshape(3,final_w*final_h)
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
		image_path_RGB = "kannala/lit/"
		image_path_S = "kannala/object_mask/"
		image_path_D = "kannala/depth_data/"
		image_path_Dc = "kannala/depth/"
		image_path_pc = "kannala/point_cloud/"
		model_path = "kannala/class/"
		image_name = "{}-kannala_brandt_{}-{}-FOV{}-{}".format(scene,system,rot1[0]+rot2,specific[1],loc)
		#Final save
		#final.save_RGB(image_path_RGB,image_name)
		#final.save_S(image_path_S,image_name)
		#final.save_D(image_path_D,image_name)
		#final.inv_depth_map(image_path_Dc,image_name)
		#final.coded_depth(image_path_Dc,image_name)
		#final.save_PC(image_path_pc,image_name)
		final.save_model(model_path,image_name)
		fichero.main(final,model_path,image_name,specific)		

