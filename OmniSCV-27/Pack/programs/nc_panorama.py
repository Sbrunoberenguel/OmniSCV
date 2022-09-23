from unrealcv import client
from PIL import Image
import cv2
import numpy as np
import functions as f
import os,time,StringIO
import setup as setup
import fichero
import UCV_programs as UCV

def set_ue_nc(vignette):
	ucv_file = open('../unrealcv.ini','w')
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

#get images on UnrealCV

def get_images(mode,vignette,threshold,code):
	rot_file = open('cube_rot.txt', 'r')
	rotation = rot_file.read().split(' ')
	for index in range(len(rotation)//3):
		angle = rotation[index*3:index*3+3]
		client.request('vset /camera/0/rotation {} {} {}'.format(int(angle[0]),int(angle[1]),int(angle[2])))
		time.sleep(0.01)	
		client.request('vget /camera/0/{} Pack/no-central/ncp/{}{}.png'.format(mode,mode,index))
		if mode == 'lit' and vignette:
			img = Image.open("no-central/ncp/{}{}.png".format(mode,index))
			img.crop((187,187,699,699)).save("no-central/ncp/{}{}.png".format(mode,index))
		elif mode == 'object_mask':
			img = Image.open('no-central/ncp/{}{}.png'.format(mode,index))
			width,height = img.size
			img.crop((width//8,0,7*width//8,height)).save("no-central/ncp/{}{}.png".format(mode,index))	
		elif mode == "depth":
			d = client.request('vget /camera/0/depth npy')
			d_img = np.load(StringIO.StringIO(d))
			f.depth_image2(d_img,"no-central/ncp/depth",index,threshold,code,vignette)
	rot_file.close()

#Main program
def main(vignette,scene,common=[2014,512,['lit'],10.0,1,'cam_rot.txt','R','No','No','No','grey'],specific=[100]):

#-----------------------------------------------------------------------------------------------
	#No central Panorama image set-up
	ue_pack = raw_input('Executable name: ')
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

	radius = specific[0]
	#-----------------------------------------------------------------------------------------------
	loc_file = open('cam_loc.txt', 'r')
	location = loc_file.read().split('\n')

	#Geometric parameters
	Nor,Rot = f.load_geom()
	R_ucv_scv = np.array([[1,0,0],[0,-1,0],[0,0,1]])

	#Init virtual environment
	if not client.isconnected():
		set_ue_nc(vignette)
		print('PLEASE WAIT UNTIL THE WINDOW OF THE VIRTUAL ENVIRONMENT CLOSE!')
		time.sleep(2)
		os.system('.././{} &'.format(ue_pack))
		time.sleep(2)
		client.connect()
	time.sleep(0.5)
	if not client.isconnected():
		print('UnrealCV server is not running')
		print('ERROR: Try again from the main menu') 
	else:
		if not vignette:
			client.request('vrun fullscreen')
			time.sleep(0.5)
			client.request('vrun r.SetRes 640x480')
		for mode in mode_list:
			if mode == 'object_mask':
				client.request('vset /camera/0/horizontal_fieldofview 106')
			elif vignette:
				client.request('vset /camera/0/horizontal_fieldofview 120')
			else:
				client.request('vset /camera/0/horizontal_fieldofview 90')
			f.folders('nc_pano',mode,0)
			#Pixel mapping
			for pos in range(len(location)-1):
				final = np.zeros((final_h,final_w,3), np.uint8)
				r,g,b = np.zeros((final_h,final_w)),np.zeros((final_h,final_w)),np.zeros((final_h,final_w))
				center_loc = np.array(location[pos].split(' '),np.float).reshape(3,1)
				if data == 'Yes' and mode == 'depth':
					f.folders('nc_pano','depth_data',pos)
					depth = np.zeros([final_w,final_h],np.float32)
				#Camera parameters
				R_view = f.camera_rotation(rot1,rot2,pos) #Rotation matrix of the viewer
				FOV = np.pi/2.0

				#Optical centers
				x,y = np.meshgrid(np.arange(final_w),np.arange(final_h))
				theta = (1.0-2*x/float(final_w))*np.pi
				phi = (0.5-y/float(final_h))*np.pi				
				cdir = np.array([(np.cos(theta)),(np.sin(theta)),(np.full_like(theta,0))]).reshape(3,final_w*final_h)
				c_dist = np.dot(R_view,cdir)*radius
				c_loc = np.dot(R_ucv_scv,center_loc)
				centro = c_loc+c_dist
				centro = np.dot(R_ucv_scv,centro[:,:final_w])
				#vectors
				ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
				vec = np.array([(cp*ct),(cp*st),(sp)]).reshape(3,final_w*final_h)
				v_abs = np.dot(R_view,vec)
				img_index = f.get_index(v_abs)
				#Acquisition
				for i in range(final_w):
					client.request('vset /camera/0/location {} {} {}'.format(centro[0,i],centro[1,i],centro[2,i]))
					get_images(mode,vignette,threshold,code)
					imagenes = f.load_img2('ncp',mode)
					if data == 'Yes' and mode == 'depth':
						depth_pool = f.depth_file2('ncp')
					im_h,im_w,ch = imagenes[0].shape
					im_w -= 1
					im_h -= 1
					fx = (im_w/2.0)/np.tan(FOV/2.0)
					fy = (im_h/2.0)/np.tan(FOV/2.0)
					K = np.array([[fx,0,im_w/2.0],[0,fy,im_h/2.0],[0,0,1]])
					#composition
					for j in range(final_h):
						aux = j*final_w+i
						index = img_index[aux]
						n,imagen,R = Nor[index],imagenes[index],Rot[index]
						p_x, p_y = f.get_pixel(v_abs[:,aux].reshape(3,1), R, K)
						color = imagen[p_y, p_x]
						r[j,i],g[j,i],b[j,i] = color[0:3]
						if data == 'Yes' and mode=='depth':
							depth[j,i] = depth_pool[index][p_y,p_x]
				final = cv2.merge((r,g,b))
				image_path = 'nc_pano/{}/'.format(mode)
				image_name = '{}-nc_pano-{}-{}-{}'.format(scene,mode,rot1[0]+rot2,pos+start)
				cv2.imwrite('{}.png'.format(image_path+image_name),final) 
				fichero.main_nc(image_path+image_name,'non-central panorama',common,specific,R_view,mode,pos)
				if data == 'Yes' and mode == 'depth':
					np.save("nc_pano/depth_data/{}.npy".format(image_name),depth)
				if show == 'Yes':
					cv2.imshow('{}'.format(image_name),final)
					cv2.waitKey(0)
					cv2.destroyAllWindows()
				if semantica=='Yes' and mode == "object_mask":
					result_path = "nc_pano/binary_masks/"
					if not os.path.exists("{}".format(result_path+image_name)):
						os.makedirs("{}".format(result_path+image_name))
					f.build_binary_mask(scene,image_path+image_name+".png","{}{}/".format(result_path,image_name))
		get_mask_list(scene)
	client.request('vrun exit')
