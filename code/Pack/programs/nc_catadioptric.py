from unrealcv import client
from PIL import Image
import cv2
import numpy as np
import math
import programs.functions as f
import os,time,io
import programs.fichero as fichero

#La resolucion para recortar a 512x512 debe ser 887x887 a 120fov
#Se cortaran en los pixeles 187-699

def set_ue_nc(vignette,forward_5):
	ucv_file = open(os.path.join(forward_5,'unrealcv.ini'),'w')
	ucv_file.write('[UnrealCV.Core] \n')
	ucv_file.write('Port=9000 \n')
	if vignette:
		ucv_file.write('Width=886 \n')
		ucv_file.write('Height=886 \n')
		ucv_file.write('FOV=120.000000 \n')
	else:
		ucv_file.write('Width=512 \n')
		ucv_file.write('Height=512 \n')
		ucv_file.write('FOV=90.000000 \n')
	ucv_file.write('EnableInput=True \n')
	ucv_file.write('EnableRightEye=False \n')
	ucv_file.close()

def get_mask_list(scene):
	mask_file = open('mask_list_{}.txt'.format(scene),'w')
	obj_list = str(client.request('vget /objects')).split(' ')
	for obj in obj_list:
		color = str(client.request('vget /object/{}/color'.format(obj)))[1:-1].split(',')
		R,G,B = color[0][2:],color[1][2:],color[2][2:]
		mask_file.write('{} '.format(obj))
		mask_file.write('{} {} {}\n'.format(R,G,B))
	mask_file.close()

#Functions
def get_param(mirror, Z_c, R_c, tau, r, Z_s, R_s):
	if mirror == 'conic':
		cot_phi = (1+r*np.tan(2*tau))/(np.tan(2*tau)-r)
		Z_r = Z_c + R_c*cot_phi
	elif mirror == 'spheric':
		Z_rel = Z_s/R_s	
		rho = np.sqrt(r**2+1)
		gamma = (-r**2*Z_rel**2 + rho**2)#*Z_rel**2
		if gamma < 0:
			cot_phi = 1/r
			Z_r = 0
		else:
			xi = np.sqrt(gamma*Z_rel**2)
			d = 2*r**2*Z_rel**4 - 2*xi*Z_rel**2 - 3*rho**2*Z_rel**2 + rho**2
			eps = (-r**2 + 1)*Z_rel**2 + 2*xi + rho**2
			dseta = 2*r**2*Z_rel**4 - 2*xi*(-r**2*Z_rel**2 + rho**2) - rho**2*(1 + Z_rel**2)
			cot_phi = -dseta/(d*r)
			Z_r = (d+eps)*Z_s/d
	else:
		print('ERROR: Wrong mirror input')
	return cot_phi,Z_r

def get_images(mode,Zr,mirror,cam,R,vignette,threshold,code,back_5):
	rot_file = open('cube_rot.txt', 'r')
	rotation = rot_file.read().split(' ')
	delta = np.array([0,0,1]).reshape(3,1)
	delta = np.dot(R,delta)
	delta *= Zr
	client.request('vset /camera/0/location {} {} {}'.format(int(cam[0])+delta[0,0],int(cam[1])-delta[1,0],int(cam[2])+delta[2,0])) 
	for index in range(len(rotation)//3):
		angle = rotation[index*3:index*3+3]
		client.request('vset /camera/0/rotation {} {} {}'.format(int(angle[0]),int(angle[1]),int(angle[2])))
		time.sleep(0.01)	
		client.request('vget /camera/0/{} {}/Pack/no-central/{}/{}{}.png'.format(mode,back_5,mirror,mode,index))
		if mode == 'lit' and vignette:
			img = Image.open("no-central/{}/{}{}.png".format(mirror,mode,index))
			img.crop((187,187,699,699)).save("no-central/{}/{}{}.png".format(mirror,mode,index))
		elif mode == 'object_mask':
			img = Image.open('no-central/{}/{}{}.png'.format(mirror,mode,index))
			width,height = img.size
			img.crop((width//8,0,7*width//8,height)).save("no-central/{}/{}{}.png".format(mirror,mode,index))	
		elif mode == "depth":
			d = client.request('vget /camera/0/depth npy')
			d_img = np.load(io.StringIO(d))
			f.depth_image2(d_img,"no-central/{}/depth".format(mirror),index,threshold,code,vignette)
	rot_file.close()

#main program
def main(vignette,scene,common=[512,512,['lit'],10.0,1,'cam_rot.txt','R','No','No','No','grey'], specific=['spheric',0.5,60,0.3]):
	#-----------------------------------------------------------------------------------------------
	#Catadioptric image set-up
	final_w = common[0]			#Image resolution: width
	final_h = common[1]			#Image resolution: height
	mode_list = common[2]		# Modes availables:["lit","object_mask","depth"]
	threshold = common[3]
	start = common[4]
	rot1 = common[5]				
	rot2 = common[6]
	show = common[7]
	semantica = common[8]
	data = common[9]
	code = common[10]

	mirror = specific[0]
	R_s = specific[1]
	tau = np.radians(specific[2])
	Z_m = specific[3]	
#-----------------------------------------------------------------------------------------------
	loc_file = open('cam_loc.txt','r')
	location = loc_file.read().split('\n')
	x_past,y_past,Zr_past = 10000,10000,1000

	if not cv2.useOptimized():
		print('Turning on the Optimizer')
		cv2.setUseOptimized(True)

	#Conic catadioptric
	R_c,Z_c,Z_s = Z_m*np.sin(2*tau),Z_m*(1-np.cos(2*tau)),Z_m + R_s

	#Geometric parameters
	Nor,Rot = f.load_geom()

	#Camera parameters
	FOV = math.pi/2.0
	f_cata = (final_w/2.0)/math.tan(FOV/2.0)
	
	#Init composition
	if not client.isconnected():
		ue_pack = input('Executable name: ')
		back_5 = '../../../..'
		forward_5 = '../LinuxNoEditor/{}/Binaries/Linux'.format(ue_pack)
		set_ue_nc(vignette,forward_5)
		print('PLEASE WAIT UNTIL THE WINDOW OF THE VIRTUAL ENVIRONMENT CLOSE!')
		time.sleep(2)
		os.system('{}/./{} &'.format(forward_5,ue_pack))
		time.sleep(2)
		client.connect()
		if not vignette:
			client.request('vrun fullscreen')
			time.sleep(0.5)
			client.request('vrun r.SetRes 640x480')

	if not client.isconnected():
		print('ERROR: Client is not connected')
		print('ERROR: Trye again from the main menu')
	else:
		#Camera images - skybox
		for mode in mode_list:
			for pos in range(len(location)-1):
				R_cam = np.array([[1,0,0],[0,-1,0],[0,0,1]])
				R_view = f.camera_rotation(rot1,rot2,pos) #Rotation matrix of the viewer
				R_world = np.dot(R_view,R_cam)
				camera_loc = location[pos].split(' ')
				final = np.zeros((final_h,final_w,3), np.uint8)
				if data == 'Yes' and mode=='depth':   
					f.folders('nc_catadioptric','depth_data',0)
					depth = np.zeros((final_w,final_h),np.float32)
				f.folders('nc_catadioptric',mode,0)
				
			#Pixel mapping
				for r in np.arange(1,final_w/2,1):
					if r < final_w/4:
						step = 0.1
					else:
						step = 0.05
					r_hat = r/f_cata
					cot_phi, Z_r = get_param(mirror, Z_c, R_c, tau, r_hat, Z_s, R_s)
					phi_cata = math.pi/2.0 - np.arctan(cot_phi)
					get_images(mode,Z_r, mirror, camera_loc, R_world, vignette, threshold, code, back_5)
					imagenes = f.load_img2(mirror,mode)
					if data == 'Yes' and mode=='depth':
							depth_pool = f.depth_file2(mirror)
					im_h,im_w,ch = imagenes[0].shape
					im_w -= 1
					im_h -= 1
					fx = (im_w/2.0)/np.tan(FOV/2.0)
					fy = (im_h/2.0)/np.tan(FOV/2.0)
					K = np.matrix('{} 0 {}; 0 {} {}; 0 0 1.0'.format(fx,im_w/2,fy,im_h/2))
					for angle in np.arange(0,360,step):
						theta_cata = np.radians(angle)
						x = int(math.floor(final_w/2 + r*np.cos(theta_cata)))
						y = int(math.floor(final_h/2 - r*np.sin(theta_cata)))
						if x==x_past and y==y_past:
							continue
						else:
							x_past = x
							y_past = y
						z_vec = np.cos(phi_cata)
						y_vec = np.sin(phi_cata)*np.sin(theta_cata)
						x_vec = np.sin(phi_cata)*np.cos(theta_cata)
						vec_cone = np.array([x_vec,y_vec,z_vec]).reshape(3,1)
						vec = np.dot(R_world,vec_cone)
						img_index = f.get_index(vec)
						img_index = int(img_index)
						n,imagen,R = Nor[img_index],imagenes[img_index],Rot[img_index]
			#Catadioptric image build
						p_x, p_y = f.get_pixel(vec, R, K)
						color = imagen[p_y, p_x]
						if data == 'Yes' and mode=='depth':
							depth[x,y] = depth_pool[img_index][p_y,p_x]
						final[y,x] = color[0:3]
						
				image_path = 'nc_catadioptric/{}/'.format(mode)
				image_name = '{}-nc_cata_{}-{}-{}-{}'.format(scene,mirror,mode,rot1[0]+rot2,pos+start)
				cv2.imwrite('{}.png'.format(image_path+image_name),final) 
				fichero.main_nc(image_path+image_name,'non-central catadioptric'+mirror,common,specific,R_view,mode,pos)
				if data == 'Yes' and mode == 'depth':
					np.save("nc_catadioptric/depth_data/{}.npy".format(image_name),depth)
				if show == 'Yes':
					cv2.imshow('{}'.format(image_name),final)
					cv2.waitKey(0)
					cv2.destroyAllWindows()
				if semantica == 'Yes' and mode == "object_mask":
					result_path = "nc_catadioptric/binary_masks/"
					if not os.path.exists("{}".format(result_path+image_name)):
						os.makedirs("{}".format(result_path+image_name))
					f.build_binary_mask(scene,image_path+image_name+".png","{}{}/".format(result_path,image_name))
		get_mask_list(scene)	
	client.request('vrun exit')
