import cv2
import graph_cuts_stereo
import numpy as np
import os
import sys

# I/O directories
input_dir = "scenes_2006"
output_dir = "./"

def get_ssd_disparity_map(left_image, right_image, window_radius):
    '''
    Basic SSD window-based correspondence
    '''
    num_rows, num_cols = left_image.shape[:2]
    window_dimensions = 1 + 2 * window_radius
    max_disparity = int(0.1 * num_cols)

    left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY).astype(int)
    left = cv2.copyMakeBorder(left, window_radius, window_radius, window_radius, window_radius, cv2.BORDER_REFLECT_101)
    right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY).astype(int)
    right = cv2.copyMakeBorder(right, window_radius, window_radius, window_radius, window_radius, cv2.BORDER_REFLECT_101)
    disparity_map = np.zeros((num_rows, num_cols))

    # Determine best correspondence for each pixel
    for row in range(num_rows):
        for col in range(num_cols):
            template_window = left[row:row + window_dimensions, col:col + window_dimensions]
            min_ssd = float('inf')
            disparity = 0

            # Search only along horizontal epipolar line
            for epipolar_col in range(max(0, col - max_disparity), col):
                proposed_window = right[row:row + window_dimensions, epipolar_col:epipolar_col + window_dimensions]
                ssd = np.einsum('ij,ij', template_window - proposed_window, template_window - proposed_window)
        
                if ssd < min_ssd:
                    min_ssd = ssd
                    disparity = np.abs(col - epipolar_col)

            # Set disparity between pixel and its correspondent
            disparity_map[row][col] = disparity

    return disparity_map

def get_gc_disparity_map(left_image, right_image):
    '''
    Energy minimization via graph cuts method
    '''
    gc_stereo_method = graph_cuts_stereo.GraphCutsStereo(left_image, right_image)
    disparity_map = gc_stereo_method.minimize_energy()

    return disparity_map

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide a scene name.')
        exit()

    # Read scene name and associated images
    scene_name = sys.argv[1]
    left_image = cv2.imread(os.path.join(input_dir, scene_name, 'view1.png'))
    right_image = cv2.imread(os.path.join(input_dir, scene_name, 'view5.png'))

    # Resize images (so runtimes fall within a reasonable range)
    num_rows, num_cols = left_image.shape[:2]
    resized_now_rows = 150      # Restrict all images to have 200 rows
    resized_num_cols = int((resized_now_rows / num_rows) * num_cols)
    resized_left_image = cv2.resize(left_image, (resized_num_cols, resized_now_rows))
    resized_right_image = cv2.resize(right_image, (resized_num_cols, resized_now_rows))

    run_ssd = False
    run_gc = True

    if run_ssd:
        # Calculate SSD disparity map
        window_radius = 4
        ssd_disp_map = get_ssd_disparity_map(resized_left_image, resized_right_image, window_radius)
        ssd_disp_map = cv2.normalize(ssd_disp_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Save SSD disparity map
        cv2.imwrite(os.path.join(output_dir, scene_name + '_ssd_disp.png'), ssd_disp_map)

    if run_gc:
        # Calculate graph cut disparity map
        gc_disp_map_left, gc_disp_map_right = get_gc_disparity_map(resized_left_image, resized_right_image)
        gc_disp_map_left = cv2.normalize(gc_disp_map_left, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gc_disp_map_right = cv2.normalize(gc_disp_map_right, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Save graph cut disparity map
        cv2.imwrite(os.path.join(output_dir, scene_name + '_gc_disp_left.png'), gc_disp_map_left)
        cv2.imwrite(os.path.join(output_dir, scene_name + '_gc_disp_right.png'), gc_disp_map_right)
