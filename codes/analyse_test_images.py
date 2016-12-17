import imageio
imageio.plugins.ffmpeg.download()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math, os
from user_functions_P1 import *

folder = '../test_images'
outputfolder = folder + '/output'
test_images = []
test_images += [each for each in os.listdir(folder) if each.endswith('.jpg') and not each.startswith('.')]


for test_image in test_images:

    test_image_name = os.path.splitext(test_image)

    img = mpimg.imread(folder+'/'+test_image)
    img_output = process_img_p(img)

    plt.figure
    plt.imshow(img_output)

    #plt.show()
    mpimg.imsave(outputfolder + '/' + test_image_name[0] + '_out.jpg', img_output)