import numpy as np
import cv2


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    return line_img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    return lines

def calculate_slope(line):

    return (line[0][3]-line[0][1])/(line[0][2]-line[0][0])

def calculate_path_lines(raw_hough_lines):

    slopes = np.zeros(raw_hough_lines.shape[0])
    i = 0

    for raw_hough_line in raw_hough_lines:
        slopes[i] = calculate_slope(raw_hough_line)
        i += 1

    left_slope = slopes[slopes>0].mean()
    right_slope = slopes[slopes<0].mean()

    x1_left = max(raw_hough_lines[slopes > 0, :, 0])

    return path_lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(img, rho, hough_thres, min_line_length, max_line_gap,
                  theta = np.pi/180, kernel_size = 5, upper_thres = 150, lower_thres = 50,
                  v_coeff = [0.6, 0.47, 0.53]):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image with lines are drawn on lanes)

    grayimg = grayscale(img)
    blurimg = gaussian_blur(grayimg, kernel_size)
    edges = canny(blurimg, lower_thres, upper_thres)

    ysize = img.shape[0]
    xsize = img.shape[1]
    vertices = np.array([[(0, ysize), (xsize * v_coeff[1], ysize * v_coeff[0]),
                          (xsize * v_coeff[2], ysize * v_coeff[0]), (xsize, ysize)]], dtype=np.int32)
    section_img = region_of_interest(edges, vertices)

    #rho = 4  # distance resolution in pixels of the Hough grid
    #min_line_length = 80  # minimum number of pixels making up a line
    #max_line_gap = 40  # maximum gap in pixels between connectable line segments

    raw_hough_lines = hough_lines(section_img, rho, theta, hough_thres, min_line_length, max_line_gap)

    path_lines = calculate_path_lines(raw_hough_lines)

    line_img = draw_lines(img, raw_hough_lines)
    line_img_ = draw_lines(img, path_lines, color=[0, 0, 255])

    result = weighted_img(line_img, img)
    result = weighted_img(line_img_, result)

    return result

def process_img_p(img):

    rho = 4  # distance resolution in pixels of the Hough grid
    hough_thres = 20
    min_line_length = 80  # minimum number of pixels making up a line
    max_line_gap = 40  # maximum gap in pixels between connectable line segments

    result = process_image(img, rho, hough_thres, min_line_length, max_line_gap)

    return result
