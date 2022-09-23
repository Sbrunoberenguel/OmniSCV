import programs.functions as f
import os,subprocess,time
import programs.setup as setup

print('\nWelcome to the image simulator OmniSCV')
print('-------------------------------------')
print('This simulator has been made in the I3A, Spain')
print('We hope you enjoy it')
print('-------------------------------------')

def main_menu():
	print('\n Main menu:')
	print('  (Ac)quisition')
	print('  (Co)mposition')
	print('  (S)cenario name')
	print('  (F)unctions')
	print('  (E)xit')
	task1 = input('\n What do you want to do?: ').upper()
	print(' \n #---------------------------#')
	return task1

def image_system_menu():
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
	order = input('\n What projection system do you want to compose?: ').upper()
	print(' \n #---------------------------#')
	return order

def extras_menu():
	print(' Extra functions menu:')
	print('  (1)-Central layout acquisition')
	print('  (2)-Non-Central layout acquisition')
	print('  (3)-Get CFL ground truth for central panorama')
	print('  (4)-Get CFL ground truth for non-central panorama')
	print('  (5)-Create a Video from frames')
	print('  (6)-Point cloud reconstruction')
	print('  (7)-Get Random rotation matrix')
	print('  (8)-Build binary mask from semantic segmentation')
	print('  Back to (M)ain menu')
	task2 = input('\n What do you want to do?: ').upper()
	print(' \n #-----------------------#')
	return task2

simulation = True
vignette = False
scene = 'OmniSCV'
list1 = ['AC','CO','S','F','E']
list2 = ['1','2','3','4','5','6','7','8','9','M']
list3 = ['1','2','3','4','5','6','7','8','M']
Error_list1 = ['0',	'Equirectangular image generation failed',
					'Cylindric image generation failed',
					'Fish eye image generation failed',
					'Catadioptric image generation failed',
					'Scaramuzzas model image generation failed',
					'Kannala-Brandt model image generation failed',
					'Non-central circular panorama image generation failed',
					'Non-central catadioptric image generation failed']

Error_list2 = ['0',	'Central layout adquisition failed',
					'Non-central layout adquisition failed',
					'Layout extraction in equirectangular panorama failed',
					'Layout extraction in non-central panorama failed',
					'Video cretion failed',
					'Point cloud reconstruction failed',
					'Random rotation generation failed',
					'WIP: Binary mask generation failed']

while simulation:
	#Main menu
	task1 = main_menu()

	if task1 in list1:
		#Image acquisition
		if task1 == 'AC':
			try:
				import programs.get_central_images as get_img
				get_img.main(scene,vignette)
				time.sleep(3)
			except:
				print('Acquisition failed. Try again. \n')
		#Change the scenario name
		elif task1 == 'S':
			scene = input('Which is the scenario\'s name?: ')
			print('The scenario\'s name is {}'.format(scene))
		#Image composition
		elif task1 == 'CO':
			try:
				order = image_system_menu()

				if order in list2:
					if order == 'M':
						continue
					default = input(' Do you want to use default parameters for image composition? (yes/no) ').capitalize()
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
			except:
				idx = int(order)
				print(Error_list1[idx])		
	#Extra functions
		elif task1 == 'F':
			task2 = extras_menu()
			import programs.extra_func as ef

			if task2 in list3:
				if 	 task2 == '1':
					import programs.UCV_programs as UCV
					UCV.get_layout(scene,vignette)
				elif task2 == '2':
					import programs.UCV_programs as UCV
					UCV.get_nc_layout(scene,vignette)
				elif task2 == '3':
					import programs.CFL_data as CFL
					CFL.get_images(scene)
				elif task2 == '4':
					import programs.CFL_data as CFL
					CFL.get_nc_targets(scene)
				elif task2 == '5':
					ef.video(scene)
				elif task2 == '6':
					pc_name = input('Point Cloud to rebuild? ')
					ef.model_3D(pc_name)
				elif task2 == '7':
					ef.randomRotations()
				elif task2 == '8':
					img_name = input('Image to segment: ')
					ef.build_binary_mask(scene,img_name)
				elif task2 == 'M':
					continue
			else:
				print('Wrong input. Try again.')

		#Exit the simulator
		elif task1 == 'E':
			simulation = False
			print('Exiting the simulator')
			print('Have a nice day \n')
	else:
		print('Wrong input. Try again.')
	print('\007') #My name is Bond, James Bond

