from PIL import Image,ImageFilter
import numpy as np
import os,time,cv2
import programs.functions as f
import math
from tqdm import tqdm

def inlist(a,gap,lista):
	aux = np.isclose(a,lista,atol=gap)
	for i in aux:
		if i.all()==True:
			return True
		else:
			continue
	return False

def targets(path,img_name,result_path,scene):
	img = cv2.imread('{}{}.png'.format(path,img_name))
	h,w,c = img.shape
	#EDGES
	edges = cv2.Canny(img,100,200)
	#CORNERS				
	corners = np.zeros((h,w))
	for i in tqdm(range(h-3)):
		for j in range(w-3):
			window = img[i:i+3,j:j+3]
			col = np.unique(window.reshape(1,-1,3),axis=1)
			count = col.shape[1]
			if count >= 3:
				corners[i,j] = 255
	#saving edge and corner maps
	edge_name = '{}EM_gt/{}_EM.jpg'.format(result_path,img_name)
	corner_name = '{}CM_gt/{}_CM.jpg'.format(result_path,img_name)
	cv2.imwrite(edge_name,edges)
	cv2.imwrite(corner_name,corners)
	if 'equirec' in result_path:
		edges = cv2.dilate(edges,np.ones(3))
		corners = cv2.dilate(corners,np.ones(3))
		cv2.imwrite(edge_name,edges)
		cv2.imwrite(corner_name,corners)
		gaussian_blur(result_path,edge_name,corner_name)
	
def gaussian_blur(result_path,edge,corner):
	edges = Image.open(edge)
	px_edges = edges.load()
	corners = Image.open(corner)
	px_corners = corners.load()
	im_w,im_h = edges.size
	sigma = max(im_w,im_h)
	blur_edge = edges.filter(ImageFilter.GaussianBlur(radius = sigma*0.0065))
	blur_corner = corners.filter(ImageFilter.GaussianBlur(radius = sigma*0.009))
	min_e,max_e = blur_edge.getextrema()
	min_c,max_c = blur_corner.getextrema()
	if max_e == 0:
		max_e = 1
		print('Max_e fail')
	if max_c == 0:
		max_c = 1
		print('Max_c fail')
	for i in range(im_w-1):
		for j in range(im_h-1):
			c_edge = blur_edge.getpixel((i,j))
			c_corner = blur_corner.getpixel((i,j))
			px_edges[i,j] = int((c_edge*255)/float(max_e))
			px_corners[i,j] = int((c_corner*255)/float(max_c))
	edges.save(edge)
	corners.save(corner)

def png2jpg(directory,result):
	lista = os.listdir(directory)
	if not os.path.exists(result):
			os.makedirs(result)
	for image in lista:
		img = Image.open(directory+image)
		img.convert("RGB").save(result+image[0:-4]+'.jpg')
			
def setCFLfolder(path):
	if not os.path.exists("{}EM_gt/".format(path)):
		os.makedirs("{}EM_gt/".format(path))
	if not os.path.exists("{}CM_gt/".format(path)):
		os.makedirs("{}CM_gt/".format(path))
	if not os.path.exists("{}VP_gt/".format(path)):
		os.makedirs("{}VP_gt/".format(path))
	if not os.path.exists("{}HL_gt/".format(path)):
		os.makedirs("{}HL_gt/".format(path))
	if not os.path.exists("{}RGB/".format(path)):
		os.makedirs("{}RGB/".format(path))

def get_nc_targets(scene):
	if not cv2.useOptimized():
		print('Turn on the Optimizer')
		cv2.setUseOptimized(True)
	path = 'nc_pano/object_mask/'
	setCFLfolder('CFL/nc_pano/')
	img_list = os.listdir(path)
	print(img_list)
	for img_name in tqdm(img_list):
		if 'txt' in img_name:
			continue
		targets(path,img_name[:-4],'CFL/nc_pano/',scene)
		


def get_images(scene):
	#-----------------------------------------------------------------------------------------------
	final_w = 1024				#Image resolution: width
	final_h = 512				#Image resolution: height
	init_loc = 0	
	num_locs = int(input('Number of locations in the scene: '))	#Number of locations to evaluate
	loc_list = [i + init_loc for i in range(num_locs)] 	#List of locations
	rot = input('Rotation from file?(yes/no): ').capitalize()
	if rot == 'Yes':
		rot1 = ('cam_rot.txt')
		rot2 = ('R')
	else:
		rot1 = (input('Set camera view direction axis (x/y/z): ').lower())
		rot2 = (input('Set direction sign (pos/neg): ').lower())
	#-----------------------------------------------------------------------------------------------
	if not cv2.useOptimized():
		print('Turn on the Optimizer')
		cv2.setUseOptimized(True)
	#Geometric parameters
	Nor,Rot = f.load_geom()
	#Camera images - cubemap
	for loc in loc_list:
		print('Composing ground truth for CFL in location {} of {}'.format(loc,loc_list[-1]))
		result_path = "CFL/equirec/test/"
		setCFLfolder(result_path)
		final = np.zeros((final_h,final_w,3), np.uint8)
		r,g,b = np.empty(final_h*final_w),np.empty(final_h*final_w),np.empty(final_h*final_w)
		image_path = "CFL/equirec/layout/"
		image_name = "{}-equirec-{}-{}".format(scene,rot1[0]+rot2,loc)
		imagenx_pos = cv2.imread(image_path+"loc{}/img{}1.png".format(loc,loc))
		imagenx_neg = cv2.imread(image_path+"loc{}/img{}2.png".format(loc,loc))
		imageny_pos = cv2.imread(image_path+"loc{}/img{}3.png".format(loc,loc))
		imageny_neg = cv2.imread(image_path+"loc{}/img{}4.png".format(loc,loc))
		imagenz_pos = cv2.imread(image_path+"loc{}/img{}5.png".format(loc,loc))
		imagenz_neg = cv2.imread(image_path+"loc{}/img{}6.png".format(loc,loc))
		imagenes=[imagenx_pos,imagenx_neg,imageny_pos,imageny_neg,imagenz_pos,imagenz_neg]
		im_h,im_w,ch = imagenx_pos.shape
		im_w -=1
		im_h -=1
		v_north = np.array([0,0,1]).reshape(1,3)
		v_south = np.array([0,0,-1]).reshape(1,3)
		#Camera parameters
		R_cam =np.array([[0,-1,0],[0,0,-1],[1,0,0]])
		R_view = f.camera_rotation(rot1,rot2,loc) #Rotation matrix of the viewer
		R_world = np.dot(R_view,R_cam)
		FOV = math.pi/2.0
		fx = (im_w/2.0)/math.tan(FOV/2.0)
		fy = (im_h/2.0)/math.tan(FOV/2.0)
		K = np.array([	[fx,0,im_w/2],
				[0,fy,im_h/2],
				[0,0,1]])
		ini = 0
		#Pixel mapping
		x,y = np.meshgrid(np.arange(final_w),np.arange(final_h))
		theta = (1.0-2*x/float(final_w))*np.pi
		phi = (0.5-y/float(final_h))*np.pi
		ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
		vec = np.array([(cp*ct),(cp*st),(sp)]).reshape(3,final_w*final_h)
		v_abs = np.dot(R_world,vec)
		d_north = np.arccos(np.dot(v_north,v_abs))
		d_south = np.arccos(np.dot(v_south,v_abs))
		d_horizon = np.arcsin(np.dot(v_north,v_abs))
		sigma_pole, sigma_horizon = 3*math.pi/180,3*math.pi/180
		exp_north = np.exp(-np.square(d_north)/np.square(2*sigma_pole))/(sigma_pole*np.sqrt(2*math.pi))
		exp_south = np.exp(-np.square(d_south)/np.square(2*sigma_pole))/(sigma_pole*np.sqrt(2*math.pi))
		exp_horizon = np.exp(-np.square(d_horizon)/np.square(2*sigma_horizon))/(sigma_horizon*np.sqrt(2*math.pi))
		poles = exp_north + exp_south
		max_pole = np.max(poles)
		max_horizon = np.max(exp_horizon)
		img_index = f.get_index(v_abs)
		for i in range(img_index.size):
			n,imagen,R = Nor[img_index[i]],imagenes[img_index[i]],Rot[img_index[i]]
			p_x, p_y = f.get_pixel(v_abs[:,i], R, K)
			color = imagen[p_y, p_x]
			r[i],g[i],b[i] = color[0:3]
		final = cv2.merge((r,g,b)).reshape(final_h,final_w,3)
		cv2.imwrite('{}.png'.format(image_path+image_name),final)
		vpoint = poles*255/max_pole
		horizon = exp_horizon*255/max_horizon
		vp_gt = vpoint.reshape(final_h,final_w)
		hl_gt = horizon.reshape(final_h,final_w)
		cv2.imwrite('{}_VP.jpg'.format(result_path+'VP_gt/'+image_name),vp_gt)
		cv2.imwrite('{}_HL.jpg'.format(result_path+'HL_gt/'+image_name),hl_gt)
		targets(image_path,image_name,result_path,scene)
			
