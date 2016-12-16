import imageio
imageio.plugins.ffmpeg.download()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math, os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from user_functions_P1 import *

folder = '../test_images'
test_images = []
test_images += [each for each in os.listdir(folder) if each.endswith('.jpg') and not each.startswith('.')]


for test_image in test_images:

    img = mpimg.imread(folder+'/'+test_image)
    img_output = process_img_p(img)

    plt.figure
    plt.imshow(img_output, cmap='gray')
    plt.show()