import programs.functions as f
import os,subprocess,time
import programs.setup as setup

print('\n Welcome to the image simulator OmniSCV')
print('-------------------------------------')
print('This simulator has been made in the I3A, Spain')
print('We hope you enjoy it')
print('-------------------------------------')

simulation = True
vignette = False
scene = 'OmniSCV'
list1 = ['AC','CO','S','F','E','M']
list2 = ['1','2','3','4','5','6','7','8','9','M']
list3 = ['1','2','3','4','5','M']

while simulation:
	print('\n Main menu:')
	print('  (Ac)quisition')
	print('  (Co)mposition')
	print('  (S)cenario name')
	print('  (F)unctions')
	print('  (E)xit')
	task1 = raw_input('\n What do you want to do?: ').upper()
	print(' \n #---------------------------#')
	#Image acquisition
	if task1 in list1:
		if task1 == 'AC':
			import programs.get_central_images as get_img
			get_img.main(scene,vignette)
			time.sleep(3)
	#Change the scenario name
		elif task1 == 'S':
			scene = raw_input('Which is the scenario\'s name?: ')
			print('The scenario\'s name is {}'.format(scene))
	#Image composition
		elif task1 == 'CO':
			print(' Central projection systems:')
			print('  (1)-Equirectangular image')
			print('  (2)-Cylindric image')
			print('  (3)-Fish eye image')
			print('  (4)-Catadioptric image')
			print('  (5)-Scaramuzza image')
			print('  (6)-Kannala-Brandt image \n')
			print(' No Central projection systems:')
			print('  (7)-No Central Panorama image')
			print('  (8)-No Central Catadiptric image \n')
			print('  Back to (M)ain menu')
			order = raw_input('\n What projection system do you want to compose?: ').upper()
			print(' \n #---------------------------#')
			if order in list2:
				if order == 'M':
					continue
				default = raw_input(' Do you want to use default parameters for image composition? (yes/no) ').capitalize()
				if default != 'Yes':
					common, specific = setup.main(order)
					print(' \n #---------------------------#')
			#Central projection systems
				if order == '1':
					import programs.equirectangular as eq
					print('Equirectangular image')
					print('--------------------- \n')
					if default == 'Yes':
						eq.main(scene)
					else:
						eq.main(scene,common,specific)
				elif order == '2':
					import programs.cylindric as cy
					print('Cylindric image')
					print('--------------- \n')	
					if default == 'Yes':
						cy.main(scene)
					else:
						cy.main(scene,common, specific)
				elif order == '3':
					import programs.fish_eye as fe
					print('Fish eye image')
					print('-------------- \n')		
					if default == 'Yes':
						fe.main(scene)
					else:
						fe.main(scene,common, specific)
				elif order == '4':
					import programs.catadioptric as ca
					print('Catadioptric image')
					print('------------------ \n')		
					if default == 'Yes':
						ca.main(scene)
					else:
						ca.main(scene,common, specific)
				elif order == '5':
					import programs.scaramuzza as sc
					print('Scaramuzza image')
					print('---------------- \n')		
					if default == 'Yes':
						sc.main(scene)
					else:
						sc.main(scene,common, specific)
				elif order == '6':
					import programs.kannala_brandt as kb
					print('Kannala-Brandt image')
					print('-------------------- \n')
					if default == 'Yes':
						kb.main(scene)
					else:
						kb.main(scene,common, specific)
			#No central projection systems
				elif order == '7':
					import programs.nc_panorama as ncp
					print('No Central panorama image')
					print('-------------------- \n')
					if default == 'Yes':
						ncp.main(vignette,scene)
						time.sleep(4)
					else:
						ncp.main(vignette,scene,common, specific)
						time.sleep(4)
				elif order == '8':
					import programs.nc_catadioptric as ncc
					print('No Central catadioptric image')
					print('-------------------- \n')
					if default == 'Yes':
						ncc.main(vignette,scene)
						time.sleep(4)
					else:
						ncc.main(vignette,scene,common,specific)
						time.sleep(4)
			else:
				print('\n ERROR: WRONG INPUT')
				print(' \n #---------------------------#')
				continue
	#Extra functions
		elif task1 == 'F':
			#Extra functions
			print(' Extra functions menu:')
			print('  (1)-Get Layout of the scene')
			print('  (2)-Create a Video from frames')
			print('  (3)-Point cloud reconstruction')
			print('  (4)-Get CFL ground truth')
			print('  Back to (M)ain menu')
			task2 = raw_input('\n What do you want to do?: ').upper()
			print(' \n #-----------------------#')
			import programs.extra_func as ef
			if task2 in list3:
				if task2 == '1':
					import programs.UCV_programs as UCV
					UCV.get_layout(scene,vignette)
				elif task2 == '2':
					ef.video(scene)
				elif task2 == '3':
					pc_name = raw_input('Which point cloud do you want to rebuild? ')
					ef.model_3D(pc_name)
				elif task2 == '4':
					import programs.CFL_data as CFL
					CFL.get_images(scene)
					form = raw_input('Adapt RGB images to CFL format? (Yes/No): ').capitalize()
					if form == 'Yes':
						directory = 'equirectangular/lit/'
						result_path = 'CFL/test/RGB/'
						CFL.png2jpg(directory,result_path)
				elif task2 == 'M':
					continue
				else:
					print('\n ERROR: WRONG INPUT')
					print('\n #---------------------------#')
					continue
			else:
				print('\n ERROR: WRONG INPUT')
				print('\n #---------------------------#')
				continue
	#Exit the simulator
		elif task1 == 'E':
			simulation = False
			print('Exiting the simulator')
			print('Have a nice day \n')

	else:
		print('\n ERROR: WRONG INPUT')
		print('\n #---------------------------#')


