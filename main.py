import cv2
import graph_cuts_stereo
import numpy as np
import os
import sys

from matplotlib import pyplot as plt
from numba import njit, cuda

# Image directory
image_dir = "images"

@njit(target_backend='cuda')
def get_ssd_disparity_map(left_image_padded, right_image_padded, num_rows, num_cols, window_radius):
    window_dimensions = 1 + 2 * window_radius
    max_disparity = int(0.15 * num_cols)
    disparity_map = np.zeros((num_rows, num_cols))

    left = left_image_padded
    right = right_image_padded

    # Determine best correspondence for each pixel
    for row in range(num_rows):
        for col in range(num_cols):
            template_window = left[row:row + window_dimensions, col:col + window_dimensions]
            min_ssd = -1
            disparity = 0

            # Search only along horizontal epipolar line
            for epipolar_col in range(max(0, col - max_disparity), col):
                proposed_window = right[row:row + window_dimensions, epipolar_col:epipolar_col + window_dimensions]
                ssd = np.sum((template_window - proposed_window) ** 2)
        
                if min_ssd == -1 or ssd < min_ssd:
                    min_ssd = ssd
                    disparity = np.abs(col - epipolar_col)

            # Set disparity between pixel and its correspondent
            disparity_map[row][col] = disparity

    return disparity_map

def get_gc_disparity_map(left_image, right_image):
    gc_stereo_method = graph_cuts_stereo.GraphCutsStereo(left_image, right_image)
    disparity_map = gc_stereo_method.minimize_energy()

    return disparity_map

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide a scene name.')
        exit()

    # Read scene name and associated images
    scene_name = sys.argv[1]
    left_image = cv2.imread(os.path.join(image_dir, scene_name, 'left.png'))
    right_image = cv2.imread(os.path.join(image_dir, scene_name, 'right.png'))
    num_rows, num_cols = left_image.shape[:2]

    run_ssd = True
    run_gc = True

    if run_ssd:
        # Calculate SSD disparity map
        window_radius = 4

        # Preprocess image
        num_rows, num_cols = left_image.shape[:2]
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY).astype(int)
        left_image_padded = cv2.copyMakeBorder(left_image, window_radius, window_radius, window_radius, window_radius, cv2.BORDER_REFLECT_101)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY).astype(int)
        right_image_padded = cv2.copyMakeBorder(right_image, window_radius, window_radius, window_radius, window_radius, cv2.BORDER_REFLECT_101)

        ssd_disp_map = get_ssd_disparity_map(left_image_padded, right_image_padded, num_rows, num_cols, window_radius)
        cv2.imwrite(os.path.join(image_dir, scene_name, 'ssd_disp.png'), ssd_disp_map)

    if run_gc:
        # Images have been modified; read again
        left_image = cv2.imread(os.path.join(image_dir, scene_name, 'left.png'))
        right_image = cv2.imread(os.path.join(image_dir, scene_name, 'right.png'))

        # Calculate graph cut disparity map
        gc_disp_map_left = get_gc_disparity_map(left_image, right_image)

        # Save graph cut disparity map
        cv2.imwrite(os.path.join(image_dir, scene_name, 'gc_disp.png'), gc_disp_map_left)
