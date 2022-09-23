def main(order):
	nc_list = ['7','8']
	c = []
	s = []
	if order in nc_list:
		c.append(input('Set final image width: '))
		c.append(input('Set final image height: '))
		aux = raw_input('Select view mode of the final image separated by commas (lit/object_mask/depth): ')
		c.append(aux.split(','))
		if 'depth' in c[2]:
			c.append(input('Set maximun depth distance in meters: '))
		else:
			c.append(10.0)
		c.append(input('Set first index: '))
		rot = raw_input('Rotation from file?(yes/no): ').capitalize()
		if rot == 'Yes':
			c.append('cam_rot.txt')
			c.append('R')
		else:
			c.append(raw_input('Set camera view direction axis (x/y/z): ').lower())
			c.append(raw_input('Set direction sign (pos/neg): ').lower())
		if order == '7':
			s.append(input('Set Radius in centimetres: '))
		elif order == '8':
			s.append(raw_input('Set type of mirror among (spheric/conic): ').lower())
			if s[0]=='spheric':
				s.append(input('Set spheric mirror radius in meters: '))
				s.append(60)
			elif s[0]=='conic':
				s.append(0.5)
				s.append(input('Set aperture angle of the conic mirror in degrees: '))
			s.append(input('Set distance between the mirror and the camera in meters: '))
		c.append(raw_input('Do you what to see the image when it\'s done? ').capitalize())
		if 'object_mask' in c[2]:
			c.append(raw_input('Do you want to build binary mask for object detection? ').capitalize())
		else:
			c.append('No')
		if 'depth' in c[2]:
			c.append(raw_input('Do you want to create a depth data file? (yes/no) ').capitalize())
			c.append(raw_input('Depth image in (grey) scale or coded?: ').lower())
		else:
			c.append('No')
			c.append('grey')

	else:
		c.append(input('Set final image width: '))
		c.append(input('Set final image height: '))
		aux = raw_input('Select view mode of the final image separated by commas (lit/object_mask/depth): ')
		c.append(aux.split(','))
		c.append(input('Set first location: '))
		c.append(input('Set number of locations: '))
		if order == '2':
			s.append(input('Set horizontal field of view in degrees for the cylindric image: '))
			s.append(input('Set vertical field of view in degrees for the cylindric image: '))	
		elif order == '3' or order == '6': 
			s.append(raw_input('Set fisheye camera type among (equiang/stereo/orth/equisol): ').lower())
			s.append(input('Set field of view in degrees for the fisheye camera: '))
		elif order == '4':
			s.append(raw_input('Set type of mirror among (para/hyper/planar): ').lower())
			s.append(input('Set distance between the mirror and the camera in meters: '))
			s.append(input('Set latus-rectum of the mirror in meters: '))
		elif order == '5':
			s.append(raw_input('Set fisheye camera (equiang/stereo/orth/equisol) or catadioptric system (hyper/para): ').lower())
			if s[-1] == 'hyper' or s[-1] == 'para':
				s.append(input('Set distance between the mirror and the camera in meters: '))
				s.append(input('Set latus-rectum of the mirror in meters: '))
			else:
				s.append(input('Set field of view in degrees for the fisheye camera: '))
		rot = raw_input('Rotation from file?(yes/no): ').capitalize()
		if rot == 'Yes':
			c.append('cam_rot.txt')
			c.append('R')
		else:
			c.append(raw_input('Set camera view direction axis (x/y/z): ').lower())
			c.append(raw_input('Set direction sign (pos/neg): ').lower())
		c.append(raw_input('Do you what to see the image when it\'s done?(yes/no): ').capitalize())
		if 'object_mask' in c[2]:
			c.append(raw_input('Do you want to build binary mask for object detection? (yes/no) ').capitalize())
		else:
			c.append('No')
		if 'depth' in c[2]:
			c.append(raw_input('Do you want to create a depth data file?(yes/no): ').capitalize())
		else:
			c.append('No')
	return c,s

