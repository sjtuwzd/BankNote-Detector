import os
from PIL import Image
from array import *
from random import shuffle

# Load from and save to
# Names = [['./training-images','train'], ['./test-images','test']]
Names = [['./training_images_single','single'], ['./training_images_continuous','continuous']]

for name in Names:
	
	data_image = array('B')
	data_label = array('B')

	FileList = []
	dire = os.listdir(name[0])
	for dirname in dire: 
		if dirname != ".DS_Store": # [1:] Excludes .DS_Store from Mac OS
			path = os.path.join(name[0],dirname)
			for filename in os.listdir(path):
				if filename.endswith(".png"):
					FileList.append(os.path.join(name[0],dirname,filename))
	shuffle(FileList) # Usefull for further segmenting the validation set

	for filename in FileList:

		label = int(filename.split('/')[2])
		print("label: ", label)

		Im = Image.open(filename)
		# Im.show()

		pixel = Im.load()
		# print("pixel: ", pixel[29, 29])
		print(Im.size)
		width, height = Im.size

		for x in range(0,width):
			for y in range(0,height):
				data_image.append(pixel[y,x])
		
		print("image:", data_image)
		print("label: ", label)
		data_label.append(label) # labels start (one unsigned byte each)

	hexval = "{0:#0{1}x}".format(len(FileList),6) # number of files in HEX

	# header for label array

	header = array('B')
	header.extend([0,0,8,1,0,0])
	header.append(int('0x'+hexval[2:][:2],16))
	header.append(int('0x'+hexval[2:][2:],16))
	
	data_label = header + data_label

	# additional header for images array
	
	if max([width,height]) <= 256:
		header.extend([0,0,0,width,0,0,0,height])
	else:
		raise ValueError('Image exceeds maximum size: 256x256 pixels');

	header[3] = 3 # Changing MSB for image data (0x00000803)
	
	data_image = header + data_image

	output_file = open(name[1]+'-images-idx3-ubyte', 'wb')
	data_image.tofile(output_file)
	output_file.close()

	output_file = open(name[1]+'-labels-idx1-ubyte', 'wb')
	data_label.tofile(output_file)
	output_file.close()

# gzip resulting files

for name in Names:
	os.system('gzip '+name[1]+'-images-idx3-ubyte')
	os.system('gzip '+name[1]+'-labels-idx1-ubyte')