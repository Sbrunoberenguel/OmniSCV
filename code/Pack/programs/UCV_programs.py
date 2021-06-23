from unrealcv import client
import os,time
import numpy as np
import programs.functions as f
import programs.nc_panorama as ncp

def set_color(count):
	margin = 51
	R,G,B = 0,0,count*margin
	while B>255 or G>255:
		if B>255:
			B -= 255
			G += margin
		if G>255:
			G -= 255
			R += margin
	return R,G,B

def hide_objs(scene):
	layout_file = open('layout_list_{}.txt'.format(scene),'w')
	target = open('layout_colors_{}.txt'.format(scene),'w')
	count = 0
	once = 0
	if not client.isconnected():
		ue_pack = input('Name of the scenario .sh: ')
		os.system('.././{} &'.format(ue_pack))
		time.sleep(5)
		client.connect()
		time.sleep(3)
	obj_list = str(client.request('vget /objects')).split(' ')

	for obj in obj_list:
		if 'Layout' in obj:
			layout_file.write('{}\n'.format(obj))
			R,G,B = set_color(count)
			if 'Layout0' in obj and once == 1:
				R,G,B = r0,g0,b0
				once += 1
			if 'Layout0' in obj and once == 0:
				r0,g0,b0 = R,G,B
				once += 1
			client.request('vset /object/{}/show'.format(obj))
			client.request('vset /object/{}/color {} {} {}'.format(obj,R,G,B))
			target.write('{} {} {}\n'.format(B,G,R))
			count +=1
		else:
			client.request('vset /object/{}/hide'.format(obj))
	layout_file.close()
	target.close()


def get_layout(scene,vignette):
	ue_pack = input('Name of the scenario .sh: ')
	start = 0
	print('PLEASE: WAIT UNTIL THE WINDOW OF THE VIRTUAL ENVIRONMENT CLOSE!')
	time.sleep(2)
	threshold = 10.0
	mode = 'object_mask'
	if not client.isconnected():
		os.system('.././{} &'.format(ue_pack))
		time.sleep(4)
		client.connect()
		time.sleep(0.5)
	if not client.isconnected():
		print('UnrealCV server is not running')
	else:
		if not vignette:
			client.request('vrun fullscreen')
			time.sleep(0.5)
			client.request('vrun r.SetRes 640x480')
		hide_objs(scene)
		client.request('vset /camera/0/horizontal_fieldofview 106')
		time.sleep(0.5)
		rot_file = open('cube_rot.txt', 'r')
		loc_file = open('cam_loc.txt', 'r')
		rotation = rot_file.read().split(' ')
		location = loc_file.read().split('\n')
		img = "img"
		for pos in range(len(location)-1):
			path = "CFL/equirec/layout/loc{}/".format(pos+start)
			if not os.path.exists(path):
				os.makedirs(path)
			loc = location[pos].split(' ')
			client.request('vset /camera/0/location {} {} {}'.format(int(loc[0]),int(loc[1]),int(loc[2])))
			for index in range(len(rotation)//3):
				angle = rotation[index*3:index*3+3]
				client.request('vset /camera/0/rotation {} {} {}'.format(int(angle[0]),int(angle[1]),int(angle[2])))
				client.request('vget /camera/0/{} Pack/{}{}{}.png'.format(mode,path+img,pos+start,1+index))					
			
			f.cubemap(path,img, pos+start,vignette)
		client.request('vrun exit')
		time.sleep(3)
		rot_file.close()
		loc_file.close()


def get_nc_layout(scene,vignette):
	ue_pack = input('Name of the scenario .sh: ')
	start = 0
	print('PLEASE: WAIT UNTIL THE WINDOW OF THE VIRTUAL ENVIRONMENT CLOSE!')
	time.sleep(2)
	common = [1024,512,['object_mask'],10.0,0,'z','pos','No','No','No','grey']
	specific = [100]
	
	if not client.isconnected():
		os.system('.././{} &'.format(ue_pack))
		time.sleep(4)
		client.connect()
		time.sleep(0.5)
	if not client.isconnected():
		print('UnrealCV server is not running')
	else:
		if not vignette:
			client.request('vrun fullscreen')
			time.sleep(0.5)
			client.request('vrun r.SetRes 640x480')
		hide_objs(scene)
		ncp.main(vignette,scene,common,specific)


