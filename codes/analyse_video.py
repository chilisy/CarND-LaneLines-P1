import imageio
imageio.plugins.ffmpeg.download()

from user_functions_P1 import *
from moviepy.editor import VideoFileClip

folder = '../'

white_output = folder+'white.mp4'
clip1 = VideoFileClip(folder+"solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_img_p) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


yellow_output = folder+'yellow.mp4'
clip2 = VideoFileClip(folder+'solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_img_p)
yellow_clip.write_videofile(yellow_output, audio=False)


challenge_output = folder + 'extra.mp4'
clip2 = VideoFileClip(folder + 'challenge.mp4')
challenge_clip = clip2.fl_image(process_img_p)
challenge_clip.write_videofile(challenge_output, audio=False)

