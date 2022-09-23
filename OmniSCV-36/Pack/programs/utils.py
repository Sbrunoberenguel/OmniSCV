import numpy as np
import cv2
import os
import pickle as pk

#Central image class

class central_image():
	def __init__(self,width=1024, height=1024):
		self.height = height
		self.width = width
		self.max_depth = 20.0
		self.cam_type = 'Central'
		
		#Cube Maps
		self.Colour_pool = []
		self.Semantic_pool = []
		self.Depth_pool = []
		
		#Vector
		self.vec = np.zeros((3,height*width))
		self.rotation = np.eye(3)
		
		#Images
		self.RGB = np.zeros((3,height*width))
		self.S = np.zeros((3,height*width))
		self.D_data = np.zeros((1,height*width))
		self.D_colour = np.zeros((3,height*width))
		self.p_cloud = np.zeros((3,height*width))

		#Metadata
		self.location = []
		self.scene = 'OmniSCV'
		
		#Scene
		self.objects = []

	def mkdir(self,path):
		if not os.path.exists(path):
		    os.makedirs(path)

	def save_RGB(self,path_RGB,name_RGB):
		self.mkdir(path_RGB)
		img = np.transpose(self.RGB).reshape(self.height,self.width,3)
		cv2.imwrite(path_RGB+name_RGB+'.png',img)
		
	def save_S(self,path_S,name_S):
		self.mkdir(path_S)
		img = np.transpose(self.S).reshape(self.height,self.width,3)
		cv2.imwrite(path_S+name_S+'.png',img)
		
	def save_D(self,path_D,name_D):
		self.mkdir(path_D)
		np.save(path_D+name_D+'.npy',self.D_data.reshape(self.height,self.width))
	
	def save_PC(self,path_PC,name_PC):
		self.mkdir(path_PC)
		mask = np.full_like(self.D_data,self.D_data<=self.max_depth)
		#mask_inv = np.full_like(mask,mask==0)
		depth = self.D_data*mask #+ self.max_depth*mask_inv
		self.p_cloud = np.transpose(np.multiply(self.vec,depth))
		np.save(path_PC + name_PC + '.npy',self.p_cloud)
	
	def coded_depth(self,path_Dc,name_Dc):
		self.mkdir(path_Dc)
		depth = self.D_data.reshape(self.height,self.width)
		lr = self.max_depth/255.0
		lg = lr/255.0
		lb = lg/255.0
		R=np.floor(depth/lr)
		aux = depth-R*lr
		G = np.floor(aux/lg)
		aux1 = aux-G*lg
		B = np.floor(aux1/lb)
		depth_img = cv2.merge((B,G,R)).reshape(self.height,self.width,3)
		cv2.imwrite(path_Dc+name_Dc+'_d{}.png'.format(int(self.max_depth)),depth_img)
		
	def inv_depth_map(self,path_iD,name_iD):
		self.mkdir(path_iD)
		d_min = max(np.amin(self.D_data),0.1)
		norm_depth = self.D_data/d_min
		inv_depth = 255/norm_depth
		cv2.imwrite(path_iD+name_iD+'_inv.png',inv_depth.reshape(self.height,self.width))
	
	def save_model(self,path_m,name_m):
		self.mkdir(path_m)
		self.Colour_pool = []
		self.Semantic_pool = []
		self.Depth_pool = []
		modelo = open(path_m+name_m+'.pkl','wb')
		pk.dump(self,modelo)
		modelo.close()
		
	

