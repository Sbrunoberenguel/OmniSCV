import os,time,cv2
from PIL import Image
import numpy as np
import StringIO

#creates folders
def folders(cam,mode,pos):
	if cam == 'central':
		if not os.path.exists("{}/{}/loc{}".format(cam,mode,pos)):
			os.makedirs("{}/{}/loc{}".format(cam,mode,pos))
	elif cam == 'no-central':
		if not os.path.exists("{}/{}".format(cam,mode)):
			os.makedirs("{}/{}".format(cam,mode))
	else:
		if not os.path.exists("{}/{}".format(cam,mode)):
			os.makedirs("{}/{}".format(cam,mode))

#UnrealCV geometry
def load_geom():
	Rx_pos = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
	nx_pos = np.array([1,0,0]).reshape(3,1)
	Rx_neg = np.array([[0,1,0],[0,0,-1],[-1,0,0]])
	nx_neg = np.array([-1,0,0]).reshape(3,1)
	Ry_pos = np.array([[1,0,0],[0,0,-1],[0,1,0]])
	ny_pos = np.array([0,1,0]).reshape(3,1)
	Ry_neg = np.array([[-1,0,0],[0,0,-1],[0,-1,0]])
	ny_neg = np.array([0,-1,0]).reshape(3,1)
	Rz_pos = np.array([[0,-1,0],[1,0,0],[0,0,1]])
	nz_pos = np.array([0,0,1]).reshape(3,1)
	Rz_neg = np.array([[0,-1,0],[-1,0,0],[0,0,-1]])
	nz_neg = np.array([0,0,-1]).reshape(3,1)
	Rot = [Rx_pos,Rx_neg,Ry_pos,Ry_neg,Rz_pos,Rz_neg]
	Nor = [nx_pos,nx_neg,ny_pos,ny_neg,nz_pos,nz_neg]
	return Nor,Rot

def load_img(mode,loc):
	imagenx_pos = cv2.imread("central/{}/loc{}/{}{}1.png".format(mode,loc,mode,loc))
	imagenx_neg = cv2.imread("central/{}/loc{}/{}{}2.png".format(mode,loc,mode,loc))
	imageny_pos = cv2.imread("central/{}/loc{}/{}{}3.png".format(mode,loc,mode,loc))
	imageny_neg = cv2.imread("central/{}/loc{}/{}{}4.png".format(mode,loc,mode,loc))
	imagenz_pos = cv2.imread("central/{}/loc{}/{}{}5.png".format(mode,loc,mode,loc))
	imagenz_neg = cv2.imread("central/{}/loc{}/{}{}6.png".format(mode,loc,mode,loc))
	imagenes=[imagenx_pos,imagenx_neg,imageny_pos,imageny_neg,imagenz_pos,imagenz_neg]
	return imagenes
	
def load_img2(system,mode):
	imagenx_pos = cv2.imread('no-central/{}/{}0.png'.format(system,mode))
	imagenx_neg = cv2.imread('no-central/{}/{}1.png'.format(system,mode))
	imageny_pos = cv2.imread('no-central/{}/{}2.png'.format(system,mode))
	imageny_neg = cv2.imread('no-central/{}/{}3.png'.format(system,mode))
	imagenz_pos = cv2.imread('no-central/{}/{}4.png'.format(system,mode))
	imagenz_neg = cv2.imread('no-central/{}/{}5.png'.format(system,mode))
	imagenes=[imagenx_pos,imagenx_neg,imageny_pos,imageny_neg,imagenz_pos,imagenz_neg]
	return imagenes
	
def inlist(a,gap,lista):
	aux = np.isclose(a,lista,atol=gap)
	for i in aux:
		if i.all()==True:
			return True
		else:
			continue
	return False

#Creates binary mask for each object
def build_binary_mask(scene,image_name,result_path):
	f = open('mask_list_{}.txt'.format(scene),'r')
	image = Image.open(image_name)
	im_w,im_h = image.size
	clist = image.getcolors()
	th_min = 1e3
	th_max = 1e8
	gap = 3
	img = Image.new("RGB",(im_w,im_h),"black")
	pixels = img.load()
	mask_file = f.read().split('\n')
	for item in range(len(mask_file)-1):
		mask = mask_file[item].split(' ')
		obj = mask[0]
		R,G,B = int(mask[1]),int(mask[2]),int(mask[3])
		R_min,R_max = R-gap,R+gap
		G_min,G_max = G-gap,G+gap
		B_min,B_max = B-gap,B+gap
		for i in range(len(clist)):
			rs,gs,bs = clist[i][1]
			if (th_min<clist[i][0]<th_max):
				if (R_min<rs<R_max) and (G_min<gs<G_max) and (B_min<bs<B_max):
					for x in range(im_w):
						for y in range(im_h):
							color = image.getpixel((x,y))
							if (R_min<color[0]<R_max) and (G_min<color[1]<G_max) and (B_min<color[2]<B_max):
								pixels[x,y] = (255,255,255)
							else:
								pixels[x,y] = (0,0,0)
					img.save('{}{}.png'.format(result_path,obj))	
					break
			else:
				continue

#creates a cubemap from 6 images
def cubemap(path, mode, pos,vignette):
	imagen1 = Image.open("{}{}{}1.png".format(path,mode,pos))
	imagen2 = Image.open("{}{}{}2.png".format(path,mode,pos))
	imagen3 = Image.open("{}{}{}3.png".format(path,mode,pos))
	imagen4 = Image.open("{}{}{}4.png".format(path,mode,pos))
	imagen5 = Image.open("{}{}{}5.png".format(path,mode,pos))
	imagen6 = Image.open("{}{}{}6.png".format(path,mode,pos))
	width, height = imagen1.size

	if mode == "lit" and vignette:	#Image resolution 1774x1774
		imagen1.crop((375,375,1399,1399)).save("{}{}{}1.png".format(path,mode,pos))
		imagen2.crop((375,375,1399,1399)).save("{}{}{}2.png".format(path,mode,pos))
		imagen3.crop((375,375,1399,1399)).save("{}{}{}3.png".format(path,mode,pos))
		imagen4.crop((375,375,1399,1399)).save("{}{}{}4.png".format(path,mode,pos))
		imagen5.crop((375,375,1399,1399)).save("{}{}{}5.png".format(path,mode,pos))
		imagen6.crop((375,375,1399,1399)).save("{}{}{}6.png".format(path,mode,pos))
	if mode == "object_mask" or  mode == "img": #Image resolution 640x480
		imagen1.crop((width//8,0,7*width//8,height)).save("{}{}{}1.png".format(path,mode,pos))
		imagen2.crop((width//8,0,7*width//8,height)).save("{}{}{}2.png".format(path,mode,pos))
		imagen3.crop((width//8,0,7*width//8,height)).save("{}{}{}3.png".format(path,mode,pos))
		imagen4.crop((width//8,0,7*width//8,height)).save("{}{}{}4.png".format(path,mode,pos))
		imagen5.crop((width//8,0,7*width//8,height)).save("{}{}{}5.png".format(path,mode,pos))
		imagen6.crop((width//8,0,7*width//8,height)).save("{}{}{}6.png".format(path,mode,pos))
	imagen1 = Image.open("{}{}{}1.png".format(path,mode,pos))
	imagen2 = Image.open("{}{}{}2.png".format(path,mode,pos))
	imagen3 = Image.open("{}{}{}3.png".format(path,mode,pos))
	imagen4 = Image.open("{}{}{}4.png".format(path,mode,pos))
	imagen5 = Image.open("{}{}{}5.png".format(path,mode,pos))
	imagen6 = Image.open("{}{}{}6.png".format(path,mode,pos))
	width, height = imagen1.size

	#Cubemap build only for show
	final = Image.new("RGBA",(4*width,3*height),(0,0,0,0))
	final.paste(imagen3,(0,height))
	final.paste(imagen1,(width,height))
	final.paste(imagen5,(width,0))
	final.paste(imagen6,(width,2*height))
	final.paste(imagen4,(2*width,height))
	final.paste(imagen2,(3*width,height))
	final.save("{}cubeMap{}{}.png".format(path,mode,pos))

#Get acquisition plane
def get_index(vec):
	ap = np.arccos(vec[0])
	am = np.arccos(-vec[0])
	bp = np.arccos(vec[1])
	bm = np.arccos(-vec[1])
	cp = np.arccos(vec[2])
	cm = np.arccos(-vec[2])
	mat = np.array([ap,am,bp,bm,cp,cm])
	return np.argmin(mat,axis=0)

#Chooses among 2 images, whichever if closer to the vector
def choose_image2(vec, img1, R1, n1, img2, R2, n2):
    u1 = vec - np.inner(n1,np.dot(np.transpose(n1),vec))
    d1 = np.sqrt(u1[0,0]**2 + u1[1,0]**2 + u1[2,0]**2)
    u2 = vec - np.inner(n2,np.dot(np.transpose(n2),vec))
    d2 = np.sqrt(u2[0,0]**2 + u2[1,0]**2 + u2[2,0]**2)
    if d1 <= d2:
	return img1, R1, n1
    else:
	return img2, R2, n2

#Takes a pixel from an image given a vector
def get_pixel(vec, rotacion, K):
	vec_img = np.dot(rotacion,vec)
	pixel = np.dot(K,vec_img)
	x = pixel[0]/pixel[2]
	y = pixel[1]/pixel[2]
	return int(x), int(y)

#Generates a .png image from depth .npy data
#Encodes de depth information into the RGB channels
def encode(depth,d_max):
	mask = np.full_like(depth,depth<=d_max)
	mask_inv = np.full_like(mask,mask==0)
	d = depth*mask+d_max*mask_inv
	lr = d_max/255.0
	lg = lr/255.0
	lb = lg/255.0
	R=np.floor(depth/lr)
	aux = depth-R*lr
	G = np.floor(aux/lg)
	aux1 = aux-G*lg
	B = np.floor(aux1/lb)
	return R,G,B

def decode(R,G,B,d_max):
	int1 = d_max/255.0
	int2 = (d_max/255.0)/255.0
	d1 = (R*d_max)/255.0
	d2 = (G/255.0)*int1
	d3 = (B/255.0)*int2
	return d1+d2+d3

#manipulates the data and creates RGB image and npy file from depth
def depth_image(depth,path,pos,index,threshold,code,vignette):
	depth_img = np.zeros((1024,1024,3), np.uint8)
	dc_img = np.load(StringIO.StringIO(depth))
	if vignette:
		d_img = dc_img[375:1399,375:1399]
	else:
		d_img = dc_img
	np.save('{}{}{}-data.npy'.format(path,pos,index),d_img) 
	if code == 'grey':
		img = (d_img/threshold)*255		
		depth_img = cv2.merge((img,img,img)).reshape(1024,1024,3)
	else:
		r,g,b = encode(d_img,threshold)
		depth_img = cv2.merge((b,g,r)).reshape(1024,1024,3)
	cv2.imwrite('{}{}{}.png'.format(path,pos,index),depth_img) 

def depth_image2(depth,path,view,threshold,code,vignette):
	print(depth.shape)
	depth_img = np.zeros((512,512,3), np.uint8)
	if vignette:
		d_img = depth[187:699,187:699]
	else:
		d_img = depth
	np.save('{}{}-data.npy'.format(path,view),d_img)
	if code == 'grey':
		img = (d_img/threshold)*255		
		depth_img = cv2.merge((img,img,img)).reshape(512,512,3)
	else:
		r,g,b = encode(d_img,threshold)
		depth_img = cv2.merge((b,g,r)).reshape(512,512,3)
	cv2.imwrite('{}{}.png'.format(path,view),depth_img) 
	
#returns depth data from a file
def depth_file(loc):
	d_xpos = np.load('central/depth/loc{}/depth{}1-data.npy'.format(loc,loc))
	d_xneg = np.load('central/depth/loc{}/depth{}2-data.npy'.format(loc,loc))
	d_ypos = np.load('central/depth/loc{}/depth{}3-data.npy'.format(loc,loc))
	d_yneg = np.load('central/depth/loc{}/depth{}4-data.npy'.format(loc,loc))
	d_zpos = np.load('central/depth/loc{}/depth{}5-data.npy'.format(loc,loc))
	d_zneg = np.load('central/depth/loc{}/depth{}6-data.npy'.format(loc,loc))
	return [d_xpos,d_xneg,d_ypos,d_yneg,d_zpos,d_zneg]
		  
#returns depth data from a file
def depth_file2(system):
	d_xpos = np.load('no-central/{}/depth0-data.npy'.format(system))
	d_xneg = np.load('no-central/{}/depth1-data.npy'.format(system))
	d_ypos = np.load('no-central/{}/depth2-data.npy'.format(system))
	d_yneg = np.load('no-central/{}/depth3-data.npy'.format(system))
	d_zpos = np.load('no-central/{}/depth4-data.npy'.format(system))
	d_zneg = np.load('no-central/{}/depth5-data.npy'.format(system))
	return [d_xpos,d_xneg,d_ypos,d_yneg,d_zpos,d_zneg]   

# Pre-made rotation matrices for camera direction
def camera_direction(c_dir,sign):
	if c_dir == 'x':
		if sign == 'pos':
			a1,a2,a3 = -90,0,-90
		else:
			a1,a2,a3 = 90,0,-90	
	elif c_dir == 'y':
		if sign == 'pos':
			a1,a2,a3 = 0,0,-90
		else:
			a1,a2,a3 = 180,0,-90	
	elif c_dir == 'z':
		if sign == 'pos':
			a1,a2,a3 = 0,0,0
		else:
			a1,a2,a3 = 0,180,0
	return a1,a2,a3

#Rotation matrix from UnrealCV angles
def camera_rotation(rotation,sign,loc):
	if rotation == 'x' or rotation == 'y' or rotation == 'z':
		a1,a2,a3 = camera_direction(rotation,sign)
		form = 'YPR'
	else:
		# Rz(a1)*Ry(a2)*Rx(a3)
		rot = open(rotation,'r').read()
		rotations = rot.split('\n')
		form = rotations[loc].split(':')[0]
		data = rotations[loc].split(':')[1]
		if form == 'RL':
			R = np.array(data.split(' '),np.float).reshape(3,3)
			return R
		elif form == 'RC':
			R = np.array(data.split(' '),np.float).reshape(3,3)
			return np.transpose(R)
		else:			
			angle = data.split(' ')
			a1 = float(angle[0])
			a2 = float(angle[1])
			a3 = float(angle[2])
	a1,a2,a3 = np.deg2rad(a1),np.deg2rad(a2),np.deg2rad(a3)
	c1,s1 = float(np.cos(a1)),float(np.sin(a1)) 
	c2,s2 = float(np.cos(a2)),float(np.sin(a2))
	c3,s3 = float(np.cos(a3)),float(np.sin(a3))
	Rz = np.array([[c1,-s1,0],[s1,c1,0],[0,0,1]])
	Ry = np.array([[c2,0,s2],[0,1,0],[-s2,0,c2]])
	Rx = np.array([[1,0,0],[0,c3,-s3],[0,s3,c3]])
	R_aux = np.dot(Rz,Ry)
	R = np.dot(R_aux,Rx)
	return R
		
def Ucv2CM(Pcv,Rcv,Ycv):
	Pcm = Pcv+90
	Ycm = Ycv
	Rcm = -Rcv
	return Pcm,Ycm,Rcm
	
def Ucv2CM(Pcm,Ycm,Rcm):
	Pcv = Pcm-90
	Ycv = Ycm
	Rcv = -Rcm
	return Pcv,Rcv,Ycv
	
def progreso(before,now,final):
	current = int(float(now/float(final))*10)
	if before != current:
		print('['+'#'*current+' '*(9-current)+']')
		before = current
	return before

