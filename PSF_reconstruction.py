import numpy as np
import glob
import os
import matplotlib.pyplot as plt
path = "D:\\40x04122024\\"
cropped_path = "D:\\40x04122024_cropped"
list_of_files = glob.glob(path+'*.bmp')           # create the list of file
x_z_shot = []
for file_name in list_of_files:
    im = plt.imread(file_name)
    cropped_file_name = cropped_path+file_name[-8:]
    # new_im = im[335:355, 1850:1870, :]
    new_im = im[1150:1250, 1250:1350, :]
    x_z_shot.append(new_im)
    plt.imsave(cropped_file_name, new_im)
x_z_shot = np.asarray(x_z_shot)

for i in range(len(x_z_shot[0])):
    new_im = x_z_shot[:, :, i, :]
    new_im = new_im[:, ::3, :]
    cropped_file_name = cropped_path+"\\"+str(i)+".bmp"
    plt.imsave(cropped_file_name, new_im)
# for i in range(len(x_z_shot[0])):
#     plt.imshow(x_z_shot[:, i, :, 0])
#     # abs = np.sqrt(x_z_shot[:, i, :, 0] ** 2 + x_z_shot[:, i, :, 1] ** 2 + x_z_shot[:, i, :, 2] ** 2)
#     # plt.imshow(abs)
#     plt.title("for the slice:"+ str(i))
#     plt.show()
