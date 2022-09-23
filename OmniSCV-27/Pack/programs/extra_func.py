import cv2 as cv
import open3d as o3d
import numpy as np

def model_3D(point_cloud):
	pc = np.load(point_cloud)
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pc)
	o3d.io.write_point_cloud('{}.ply'.format(point_cloud[0:-4]),pcd)
	o3d.visualization.draw_geometries([pcd])

def video(scene):
	rot_file = open('cam_rot.txt', 'r')
	loc_file = open('cam_loc.txt', 'r')
	rotation = rot_file.read().split('\n')
	location = loc_file.read().split('\n')
#Video Set-up
	pano = raw_input('Set kind of omnidirectional image for the video: ').lower()
	frames = input('Number of frames: ')
	mode = raw_input('Set image mode lit-object_mask: ')
	name = mode[0:3]
	view = raw_input('What is the view direction? (ej: xpos): ') 
	SO = raw_input('Operative system?: ').capitalize()

#Video building
	pano_video = []
	for i in range(frames):
		pano_img = cv2.imread('{}/{}/{}-{}-{}-{}.png'.format(pano,mode,scene,pano,view,i+1))
		pano_video.append(pano_img)

	hp,wp,lp = pano_video[0].shape

	if SO == 'Linux':
		out_pano = cv2.VideoWriter('{}_{}_video.avi'.format(pano,name),cv2.VideoWriter_fourcc(*"MJPG"),30.0,(wp,hp))
	else:
		out_pano = cv2.VideoWriter('{}_{}_video.mp4'.format(pano,name),cv2.VideoWriter_fourcc(*"mp4v"),30.0,(wp,hp))
		
	for i in range(frames):
		out_pano.write(pano_video[i])

	out_pano.release()


