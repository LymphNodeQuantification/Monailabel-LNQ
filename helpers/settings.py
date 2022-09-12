import os
import socket

folders = list()
host = socket.gethostname()

if host == 'hermes':
    project_root = '/media/mehrtash/lnq/lnq/'
elif host == 'bezd':
    project_root = '/media/mehrtash/lnq/lnq/'
elif host == 'LAPTOP-I517BUR0':
    project_root = 'C://Users//alire//data//lnq//'
elif host == 'PHS026879':
    project_root = '/Users/rk588/data/lnq/'
slicer_folder = os.path.join(project_root, 'slicer')

# raw folder is where we keep the original data which will be untouched
raw_folder = os.path.join(project_root, 'raw')
folders.append(raw_folder)

intermediate_folder = os.path.join(project_root, 'intermediate')
folders.append(intermediate_folder)

data_folder = os.path.join(intermediate_folder, 'data')
folders.append(data_folder)
sheets_folder = os.path.join(data_folder, 'sheets')
folders.append(sheets_folder)

#
images_folder = os.path.join(data_folder, 'images')
folders.append(images_folder)
#
arrays_folder = os.path.join(data_folder, 'arrays')
folders.append(arrays_folder)

# models folder is where we store trained deep learning models
models_folder = os.path.join(intermediate_folder, 'models')
folders.append(models_folder)

snaps_folder = os.path.join(intermediate_folder, 'snaps')
folders.append(snaps_folder)

if __name__ == '__main__':
    for folder in folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)
        else:
            print('folder {} exists.'.format(folder))
