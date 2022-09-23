from unrealcv import client
import os,time
import numpy as np
import functions as f

def hide_objs(scene):
	layout_file = open('layout_list_{}.txt'.format(scene),'w')
	target = open('layout_colors_{}.txt'.format(scene),'w')
	obj_list = str(client.request('vget /objects')).split(' ')
	for obj in obj_list:
		if 'layout' in obj:
			layout_file.write('{}\n'.format(obj))
			color = str(client.request('vget /object/{}/color'.format(obj)))[1:-1].split(',')
			R,G,B = color[0][2:],color[1][2:],color[2][2:]
			target.write('{} {} {}\n'.format(R,G,B))
		elif  'aux' in obj:
			continue
		else:
			client.request('vset /object/{}/hide'.format(obj))
	layout_file.close()
	target.close()

def get_layout(scene,vignette):
	ue_pack = raw_input('Name of the scenario .sh: ')
	start = input('Introduce starting index: ')
	print('PLEASE: WAIT UNTIL THE WINDOW OF THE VIRTUAL ENVIRONMENT CLOSE!')
	time.sleep(2)
	threshold = 10.0
	code = 'grey'
	mode = 'object_mask'
	os.system('.././{} &'.format(ue_pack))
	time.sleep(5)
	client.connect()
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
			path = "CFL/layout/loc{}/".format(pos+start)
			if not os.path.exists("{}/{}/loc{}".format('CFL','layout',pos+start)):
				os.makedirs("{}/{}/loc{}".format('CFL','layout',pos+start))
			loc = location[pos].split(' ')
			client.request('vset /camera/0/location {} {} {}'.format(int(loc[0]),int(loc[1]),int(loc[2])))
			for index in range(len(rotation)//3):
				angle = rotation[index*3:index*3+3]
				client.request('vset /camera/0/rotation {} {} {}'.format(int(angle[0]),int(angle[1]),int(angle[2])))
				if mode == "depth":	
					d_npy = client.request('vget /camera/0/depth npy')
					f.depth_image(d_npy,path+img,pos+start,1+index,threshold,code,vignette)
				else:
					client.request('vget /camera/0/{} Pack/{}{}{}.png'.format(mode,path+img,pos+start,1+index))					
			
			f.cubemap(path,img, pos+start,vignette)
		client.request('vrun exit')
		time.sleep(3)
		rot_file.close()
		loc_file.close()


