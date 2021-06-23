def main(order):
	nc_list = ['7','8']
	c = []
	s = []
	if order in nc_list:
		c.append(int(input('Set final image width: ')))
		c.append(int(input('Set final image height: ')))
		aux = input('Select view mode of the final image separated by commas (lit/object_mask/depth): ')
		c.append(aux.split(','))
		if 'depth' in c[2]:
			c.append(input('Set maximun depth distance in meters: '))
		else:
			c.append(20.0)
		c.append(int(input('Set first index: ')))
		rot = input('Rotation from file?(yes/no): ').capitalize()
		if rot == 'Yes':
			c.append('cam_rot.txt')
			c.append('R')
		else:
			c.append(input('Set system revolution axis (x/y/z): ').lower())
			c.append(input('Set direction sign (pos/neg): ').lower())
		if order == '7':
			s.append(int(input('Set Radius in centimetres: ')))
		elif order == '8':
			s.append(input('Set type of mirror among (spheric/conic): ').lower())
			if s[0]=='spheric':
				s.append(float(input('Set spheric mirror radius in meters: ')))
				s.append(60)
			elif s[0]=='conic':
				s.append(0.5)
				s.append(float(input('Set aperture angle of the conic mirror in degrees: ')))
			s.append(float(input('Set distance between the mirror and the camera in meters: ')))
		c.append(input('Do you what to see the image when it\'s done? ').capitalize())
		if 'object_mask' in c[2]:
			c.append(input('Do you want to build binary mask for object detection? ').capitalize())
		else:
			c.append('No')
		if 'depth' in c[2]:
			c.append(input('Do you want to create a depth data file? (yes/no) ').capitalize())
			c.append(input('Depth image in (grey) scale or (coded)?: ').lower())
		else:
			c.append('No')
			c.append('grey')

	else:
		c.append(int(input('Set final image width: ')))
		c.append(int(input('Set final image height: ')))
		c.append(int(input('Set first location: ')))
		c.append(int(input('Set number of locations: ')))
		if order == '2':
			s.append(int(input('Set horizontal field of view in degrees for the cylindric image: ')))
			s.append(int(input('Set vertical field of view in degrees for the cylindric image: ')))
		elif order == '3' or order == '6': 
			s.append(input('Set fisheye camera type among (equiang/stereo/orth/equisol): ').lower())
			s.append(int(input('Set field of view in degrees for the fisheye camera: ')))
		elif order == '4':
			s.append(input('Set type of mirror among (para/hyper/planar): ').lower())
			s.append(float(input('Set distance between the mirror and the camera in meters: ')))
			s.append(float(input('Set latus-rectum of the mirror in meters: ')))
		elif order == '5':
			params = input('Set intrinsic calibration parameters of Scaramuzza model separated by spaces: ').split(' ')
			for i in range(len(params)):
				s.append(float(params[i]))
			'''
			s.append(input('Set fisheye camera (equiang/stereo/orth/equisol) or catadioptric system (hyper/para): ').lower())
			if s[-1] == 'hyper' or s[-1] == 'para':
				s.append(float(input('Set distance between the mirror and the camera in meters: ')))
				s.append(float(input('Set latus-rectum of the mirror in meters: ')))
			else:
				s.append(int(input('Set field of view in degrees for the fisheye camera: ')))
			'''
		rot = input('Rotation from file?(yes/no): ').capitalize()
		if rot == 'Yes':
			c.append('cam_rot.txt')
			c.append('R')
		else:
			c.append(input('Set camera view direction axis (x/y/z): ').lower())
			c.append(input('Set direction sign (pos/neg): ').lower())
		c.append(float(input('Set maximum depth: ')))
	return c,s

