from unrealcv import client
from PIL import Image
from programs.functions import *
import numpy as np
import os,time

def set_UCV(vignette,forward_5):
	ucv_file = open(os.path.join(forward_5,'unrealcv.ini'),'w')
	ucv_file.write('[UnrealCV.Core] \n')
	ucv_file.write('Port=9000 \n')
	if vignette:
		ucv_file.write('Width=1774 \n')
		ucv_file.write('Height=1774 \n')
		ucv_file.write('FOV=120.000000 \n')
	else:
		ucv_file.write('Width=1024 \n')
		ucv_file.write('Height=1024 \n')
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

def show_hiden():
	obj_list = str(client.request('vget /objects')).split(' ')
	for obj in obj_list:
		if 'Layout' in obj.capitalize():
			client.request('vset /object/{}/show'.format(obj))

def hide_hiden():
	obj_list = str(client.request('vget /objects')).split(' ')
	for obj in obj_list:
		if 'Layout' in obj.capitalize():
			client.request('vset /object/{}/hide'.format(obj))


def main(scene,vignette):
#Set-up
	ue_pack = input('Name of the scenario .sh: ')
	back_5 = '../../../..'
	forward_5 = '../LinuxNoEditor/{}/Binaries/Linux'.format(ue_pack)
	set_UCV(vignette,forward_5)
	cube_file = open('cube_rot.txt', 'r')
	loc_file = open('cam_loc.txt', 'r')
	cube = cube_file.read().split(' ')
	location = loc_file.read().split('\n')
	mode_list = ["lit","object_mask","depth"]
	start = int(input('Introduce starting index: '))
	print('PLEASE: WAIT UNTIL THE WINDOW OF THE VIRTUAL ENVIRONMENT CLOSE!')
	time.sleep(2)

#Start virtual environment
	os.system('{}/./{} &'.format(forward_5,ue_pack))
	time.sleep(5)
	client.connect()
	if not client.isconnected():
		print('UnrealCV server is not running')
		print('Something went wrong. \n Try again from the main menu please')
	else:
	
#Image Acquisition
#		if not vignette:
		client.request('vrun fullscreen')
		time.sleep(0.5)
		client.request('vrun r.SetRes 640x480')
		hiden = True
		for mode in mode_list:
			if mode == 'lit' and not hiden:
				hide_hiden()
				hiden = False
			elif mode=='object_mask' and hiden:
				show_hiden()
				hiden = True
			elif mode=='depth' and hiden:
				show_hiden()
				hiden = True
			for pos in range(len(location)-1):
				loc = location[pos].split(' ')
				client.request('vset /camera/0/location {} {} {}'.format(float(loc[0]),float(loc[1]),float(loc[2])))			
				folders('central',mode,pos+start)
				path = "central/{}/loc{}/".format(mode,pos+start)
				if mode == 'object_mask':
					client.request('vset /camera/0/horizontal_fieldofview 106')
				elif mode == 'lit' and vignette:
					client.request('vset /camera/0/horizontal_fieldofview 120')
				else:
					client.request('vset /camera/0/horizontal_fieldofview 90')
				time.sleep(0.1)	
				for index in range(len(cube)//3):
					angle = cube[index*3:index*3+3]
					r1,r2,r3 = int(angle[0]),int(angle[1]),int(angle[2])
					client.request('vset /camera/0/rotation {} {} {}'.format(r1,r2,r3))
					time.sleep(0.05)
					if mode == "depth":	
						d_npy = client.request('vget /camera/0/depth npy')
						depth_image(d_npy,path+mode,pos+start,1+index,vignette)
					else:	    	
						client.request('vget /camera/0/{} {}/Pack/{}{}{}{}.png'.format(mode,back_5,path,mode,pos+start,1+index))
				#Cube-map building
				if mode != 'depth':
					cubemap(path, mode, pos+start,vignette)
		cube_file.close()
		loc_file.close()
	get_mask_list(scene)
	client.request('vrun exit')
	
