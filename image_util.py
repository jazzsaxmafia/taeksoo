import cv2
import numpy as np
import pylab
import math
#from taeksoo.cnn_util import crop_image


image_path = '/home/taeksoo/Study/test.jpg'
image = cv2.imread(image_path)

def largest_rotated_rect(w, h, angle):
     """
     Given a rectangle of size wxh that has been rotated by 'angle' (in
     radians), computes the width and height of the largest possible
     axis-aligned rectangle within the rotated rectangle.

     Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

     Converted to Python by Aaron Snoswell
     """

     quadrant = int(math.floor(angle / (math.pi / 2))) & 3
     sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
     alpha = (sign_alpha % math.pi + math.pi) % math.pi

     bb_w = w * math.cos(alpha) + h * math.sin(alpha)
     bb_h = w * math.sin(alpha) + h * math.cos(alpha)

     gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

     delta = math.pi - alpha - gamma

     length = h if (w < h) else w

     d = length * math.cos(alpha)
     a = d * math.sin(alpha) / math.sin(delta)

     y = a * math.cos(gamma)
     x = y * math.tan(gamma)

     return (
         bb_w - 2 * x,
         bb_h - 2 * y
     )

def crop_around_center(image, width, height):
     """
     Given a NumPy / OpenCV 2 image, crops it to the given width and height,
     around it's centre point
     """

     image_size = (image.shape[1], image.shape[0])
     image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

     if(width > image_size[0]):
         width = image_size[0]

     if(height > image_size[1]):
         height = image_size[1]

     x1 = int(image_center[0] - width * 0.5)
     x2 = int(image_center[0] + width * 0.5)
     y1 = int(image_center[1] - height * 0.5)
     y2 = int(image_center[1] + height * 0.5)

     return image[y1:y2, x1:x2]


def crop_resize(image, target_height=227, target_width=227):

     if len(image.shape) == 2:
         image = np.tile(image[:,:,None], 3)
     elif len(image.shape) == 4:
         image = image[:,:,:,0]

     height, width, rgb = image.shape
     if width == height:
         resized_image = cv2.resize(image, (target_height,target_width))

     elif height < width:
         resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
         cropping_length = int((resized_image.shape[1] - target_height) / 2)
         resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

     else:
         resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
         cropping_length = int((resized_image.shape[0] - target_width) / 2)
         resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

     return cv2.resize(resized_image, (target_height, target_width))


def blur(image):
    blur = cv2.blur(image, (9,9))
    return blur

def crop(image, src_shape=(500,500), dst_shape=(227,227)):

    image = crop_resize(image, target_height=src_shape[0], target_width=src_shape[1])

    candidate_y = np.random.choice((src_shape[0] - dst_shape[0])/2)
    candidate_x = np.random.choice((src_shape[1] - dst_shape[1])/2)

    return image[candidate_y:candidate_y+dst_shape[0],candidate_x:candidate_x+dst_shape[1]]

def rotate(image, degree=45):
    rows, cols, channel = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), degree, 1)

    rotated = cv2.warpAffine(image, M, (cols, rows))
    rot_rect = crop_around_center(
            rotated,
            *largest_rotated_rect(cols, rows, math.radians(degree)))

    return rot_rect






