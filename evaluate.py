import cv2
import numpy as np
import os

from matplotlib import pyplot as plt

# Image directory
image_dir = "images"

def proportion_disparities_in_range(true_disp, disp_hat, range_value=1):
    nonzero_mask = disp_hat != 0
    nonzero_range_mask = (np.abs(true_disp - disp_hat) <= range_value) & (disp_hat != 0)
    return np.sum(nonzero_range_mask) / np.sum(nonzero_mask)

def generate_proportion_plot(scene_name, true_disp, ssd_disp, gc_disp):
    thresh_values = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60]
    ssd_proportions = [proportion_disparities_in_range(true_disp, ssd_disp, thresh_value) for thresh_value in thresh_values]
    gc_proportions = [proportion_disparities_in_range(true_disp, gc_disp, thresh_value) for thresh_value in thresh_values]
    plt.plot(thresh_values, ssd_proportions, label='Window-Based Matching')
    plt.plot(thresh_values, gc_proportions, label='Energy Minimization')

    plt.rcParams.update({'font.size': 12})
    plt.xscale('log')
    plt.xlabel('Threshold (Log-Scaled)')
    plt.ylabel('Proportion')
    plt.title(scene_name + ': Proportion of Disparities \n That Fall Within Threshold of Ground Truth')
    plt.legend()
    plt.savefig(os.path.join(image_dir, scene_name, 'proportion_plot.png'))
    plt.clf()

if __name__ == '__main__':
    
    #################### Evaluate Aloe ####################
    aloe_disp = cv2.imread(os.path.join(image_dir, 'Aloe/true_disp.png')).astype(int)
    aloe_ssd_disp = cv2.imread(os.path.join(image_dir, 'Aloe/ssd_disp.png')).astype(int)
    aloe_gc_disp = cv2.imread(os.path.join(image_dir, 'Aloe/gc_disp.png')).astype(int)

    plt.imsave(os.path.join(image_dir, 'Aloe/gray_true_disp.png'), aloe_disp.astype(np.uint8))
    plt.imsave(os.path.join(image_dir, 'Aloe/gray_ssd_disp.png'), aloe_ssd_disp.astype(np.uint8))
    plt.imsave(os.path.join(image_dir, 'Aloe/gray_gc_disp.png'), aloe_gc_disp.astype(np.uint8))
    plt.imsave(os.path.join(image_dir, 'Aloe/color_true_disp.png'), aloe_disp.astype(np.uint8)[:, :, 0], cmap="jet")
    plt.imsave(os.path.join(image_dir, 'Aloe/color_ssd_disp.png'), aloe_ssd_disp.astype(np.uint8)[:, :, 0], cmap="jet")
    plt.imsave(os.path.join(image_dir, 'Aloe/color_gc_disp.png'), aloe_gc_disp.astype(np.uint8)[:, :, 0], cmap="jet")

    generate_proportion_plot('Aloe', aloe_disp, aloe_ssd_disp, aloe_gc_disp)

    aloe_ssd_nonzero_mask = aloe_ssd_disp != 0
    aloe_ssd_nonzero_mae = np.abs(aloe_disp[aloe_ssd_nonzero_mask] - aloe_ssd_disp[aloe_ssd_nonzero_mask]).mean()
    aloe_gc_nonzero_mask = aloe_gc_disp != 0
    aloe_gc_nonzero_mae = np.abs(aloe_disp[aloe_gc_nonzero_mask] - aloe_gc_disp[aloe_gc_nonzero_mask]).mean()
    print('SSD Nonzero MAE: ', aloe_ssd_nonzero_mae)
    print('GC Nonzero MAE: ', aloe_gc_nonzero_mae)

    #################### Evaluate Baby ####################
    baby_disp = cv2.imread(os.path.join(image_dir, 'Baby/true_disp.png')).astype(int)
    baby_ssd_disp = cv2.imread(os.path.join(image_dir, 'Baby/ssd_disp.png')).astype(int)
    baby_gc_disp = cv2.imread(os.path.join(image_dir, 'Baby/gc_disp.png')).astype(int)

    plt.imsave(os.path.join(image_dir, 'Baby/gray_true_disp.png'), baby_disp.astype(np.uint8))
    plt.imsave(os.path.join(image_dir, 'Baby/gray_ssd_disp.png'), baby_ssd_disp.astype(np.uint8))
    plt.imsave(os.path.join(image_dir, 'Baby/gray_gc_disp.png'), baby_gc_disp.astype(np.uint8))
    plt.imsave(os.path.join(image_dir, 'Baby/color_true_disp.png'), baby_disp.astype(np.uint8)[:, :, 0], cmap="jet")
    plt.imsave(os.path.join(image_dir, 'Baby/color_ssd_disp.png'), baby_ssd_disp.astype(np.uint8)[:, :, 0], cmap="jet")
    plt.imsave(os.path.join(image_dir, 'Baby/color_gc_disp.png'), baby_gc_disp.astype(np.uint8)[:, :, 0], cmap="jet")

    generate_proportion_plot('Baby', baby_disp, baby_ssd_disp, baby_gc_disp)

    baby_ssd_nonzero_mask = baby_ssd_disp != 0
    baby_ssd_nonzero_mae = np.abs(baby_disp[baby_ssd_nonzero_mask] - baby_ssd_disp[baby_ssd_nonzero_mask]).mean()
    baby_gc_nonzero_mask = baby_gc_disp != 0
    baby_gc_nonzero_mae = np.abs(baby_disp[baby_gc_nonzero_mask] - baby_gc_disp[baby_gc_nonzero_mask]).mean()
    print('SSD Nonzero MAE: ', baby_ssd_nonzero_mae)
    print('GC Nonzero MAE: ', baby_gc_nonzero_mae)

    #################### Evaluate Bowling ####################
    bowling_disp = cv2.imread(os.path.join(image_dir, 'Bowling/true_disp.png')).astype(int)
    bowling_ssd_disp = cv2.imread(os.path.join(image_dir, 'Bowling/ssd_disp.png')).astype(int)
    bowling_gc_disp = cv2.imread(os.path.join(image_dir, 'Bowling/gc_disp.png')).astype(int)

    plt.imsave(os.path.join(image_dir, 'Bowling/gray_true_disp.png'), bowling_disp.astype(np.uint8))
    plt.imsave(os.path.join(image_dir, 'Bowling/gray_ssd_disp.png'), bowling_ssd_disp.astype(np.uint8))
    plt.imsave(os.path.join(image_dir, 'Bowling/gray_gc_disp.png'), bowling_gc_disp.astype(np.uint8))
    plt.imsave(os.path.join(image_dir, 'Bowling/color_true_disp.png'), bowling_disp.astype(np.uint8)[:, :, 0], cmap="jet")
    plt.imsave(os.path.join(image_dir, 'Bowling/color_ssd_disp.png'), bowling_ssd_disp.astype(np.uint8)[:, :, 0], cmap="jet")
    plt.imsave(os.path.join(image_dir, 'Bowling/color_gc_disp.png'), bowling_gc_disp.astype(np.uint8)[:, :, 0], cmap="jet")

    generate_proportion_plot('Bowling', bowling_disp, bowling_ssd_disp, bowling_gc_disp)

    bowling_ssd_nonzero_mask = bowling_ssd_disp != 0
    bowling_ssd_nonzero_mae = np.abs(bowling_disp[bowling_ssd_nonzero_mask] - bowling_ssd_disp[bowling_ssd_nonzero_mask]).mean()
    bowling_gc_nonzero_mask = bowling_gc_disp != 0
    bowling_gc_nonzero_mae = np.abs(bowling_disp[bowling_gc_nonzero_mask] - bowling_gc_disp[bowling_gc_nonzero_mask]).mean()
    print('SSD Nonzero MAE: ', bowling_ssd_nonzero_mae)
    print('GC Nonzero MAE: ', bowling_gc_nonzero_mae)
