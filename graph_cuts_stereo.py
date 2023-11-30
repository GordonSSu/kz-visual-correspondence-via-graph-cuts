import cv2
import maxflow
import numpy as np
import random

import matplotlib.pyplot as plt

VAR_ALPHA = -1
VAR_ABSENT = -2
NEIGHBORS = [[0, 1], [1, 0]]

class GraphCutsStereo:
    def __init__(self, left_image, right_image):
        self.num_rows, self.num_cols = left_image.shape[:2]
        self.left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY).astype(int)
        self.right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY).astype(int)

        self.max_disparity = int(0.12 * self.num_cols)
        self.I_disp = list(range(1, self.max_disparity))
        self.done = [False for i in range(len(self.I_disp))]

        window_radius = 3
        self.d_left = np.full((self.num_rows, self.num_cols), -1) # np.zeros((self.num_rows, self.num_cols), dtype=int)
        self.d_right = np.full((self.num_rows, self.num_cols), -1)
        # self.initialize_d_left_right(window_radius)

        self.K = 30
        self.lam = 8
        self.curr_energy = float('inf')
        self.constant_additional_cut_cost = 0
        self.vars_A = np.zeros((self.num_rows, self.num_cols), dtype=int)
        self.vars_0 = np.zeros((self.num_rows, self.num_cols), dtype=int)

    def initialize_d_left_right(self, window_radius):
        window_dimensions = 1 + 2 * window_radius

        left = cv2.copyMakeBorder(self.left_image, window_radius, window_radius, window_radius, window_radius, cv2.BORDER_REFLECT_101)
        right = cv2.copyMakeBorder(self.right_image, window_radius, window_radius, window_radius, window_radius, cv2.BORDER_REFLECT_101)

        # Determine best correspondence for each pixel
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                template_window = left[row:row + window_dimensions, col:col + window_dimensions]
                min_ssd = float('inf')
                disparity = 0

                # Search only along horizontal epipolar line
                for epipolar_col in range(max(0, col - self.max_disparity), col):
                    proposed_window = right[row:row + window_dimensions, epipolar_col:epipolar_col + window_dimensions]
                    ssd = np.einsum('ij,ij', template_window - proposed_window, template_window - proposed_window)
            
                    if ssd < min_ssd:
                        min_ssd = ssd
                        disparity = col - epipolar_col

                # Set disparity between pixel and its correspondent
                self.d_left[row][col] = disparity
                self.d_right[row][col - disparity] = disparity

    def reset_state(self):
        self.constant_additional_cut_cost = 0
        self.vars_A = np.zeros((self.num_rows, self.num_cols))
        self.vars_0 = np.zeros((self.num_rows, self.num_cols))

    def add_nodes(self, g, alpha):
        num_nodes = 0

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                
                # Pixel is unoccluded
                if self.d_left[row][col] >= 0:

                    # Disparity == alpha
                    if self.d_left[row][col] == alpha:
                        self.vars_A[row][col] = VAR_ALPHA
                        self.vars_0[row][col] = VAR_ALPHA

                    # Disparity != alpha
                    else:
                        self.vars_0[row][col] = num_nodes
                        num_nodes += 1

                        # Pixel + alpha exists in right image
                        if col >= alpha:
                            self.vars_A[row][col] = num_nodes
                            num_nodes += 1

                        # Pixel + alpha does not exist in right image
                        else:
                            self.vars_A[row][col] = VAR_ABSENT

                # Pixel is occluded
                else:
                    self.vars_0[row][col] = VAR_ABSENT

                    # Pixel + alpha exists in right image
                    if col >= alpha:
                        self.vars_A[row][col] = num_nodes
                        num_nodes += 1

                    # Pixel + alpha does not exist in right image
                    else:
                        self.vars_A[row][col] = VAR_ABSENT

        nodes = g.add_nodes(num_nodes)
        return nodes

    def create_data_occlusion_edges(self, g, alpha):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                
                # Currently active assigment with disparity == alpha
                if self.vars_A[row][col] == VAR_ALPHA and self.vars_0[row][col] == VAR_ALPHA:
                    D_prime = np.abs(self.left_image[row][col] - self.right_image[row][col - alpha]) ** 2 - self.K
                    self.constant_additional_cut_cost += D_prime

                # Currently inactive assigment with disparity == alpha
                if self.vars_A[row][col] >= 0:
                    a = self.vars_A[row][col]
                    D_prime = np.abs(self.left_image[row][col] - self.right_image[row][col - alpha]) ** 2 - self.K
                    g.add_tedge(a, D_prime, 0)

                # Currently active assigment with disparity != alpha
                if self.vars_0[row][col] >= 0:
                    o = self.vars_0[row][col]
                    pixel_disparity = self.d_left[row][col]
                    D_prime = np.abs(self.left_image[row][col] - self.right_image[row][col - pixel_disparity]) ** 2 - self.K
                    g.add_tedge(o, 0, D_prime)

    def add_term2(self, g, a1, a2, E_00, E_01, E_10, E_11):
        g.add_tedge(a1, E_11, E_01)
        g.add_tedge(a2, 0, E_00 - E_01)
        g.add_edge(a2, a1, E_01 + E_10 - E_00 - E_11, 0)

    def create_smoothness_edges(self, g, alpha):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                for neighbor in NEIGHBORS:
                    neighbor_row = row + neighbor[0]
                    neighbor_col = col + neighbor[1]
                    if neighbor_row < self.num_rows and neighbor_col < self.num_cols:
                        
                        # Case 1: Alpha disparity neighbors: both inactive
                        if self.vars_A[row][col] >= 0 and self.vars_A[neighbor_row][neighbor_col] >= 0:
                            a1 = self.vars_A[row][col]
                            a2 = self.vars_A[neighbor_row][neighbor_col]
                            V = self.lam
                            
                            if max(np.abs(self.left_image[row][col] - self.left_image[neighbor_row][neighbor_col]), \
                                np.abs(self.right_image[row][col - alpha] - self.right_image[neighbor_row][neighbor_col - alpha])) < 8:
                                V = 3 * self.lam

                            self.add_term2(g, a1, a2, 0, V, V, 0)
                            # self.add_term2(g, a2, a1, 0, V, V, 0)

                        # Case 2: Alpha disparity neighbors: one inactive
                        if self.vars_A[row][col] >= 0 and self.vars_A[neighbor_row][neighbor_col] == VAR_ALPHA:
                            a1 = self.vars_A[row][col]
                            V = self.lam

                            if max(np.abs(self.left_image[row][col] - self.left_image[neighbor_row][neighbor_col]), \
                                np.abs(self.right_image[row][col - alpha] - self.right_image[neighbor_row][neighbor_col - alpha])) < 8:
                                V = 3 * self.lam

                            g.add_tedge(a1, 0, V)
                        elif self.vars_A[row][col] == VAR_ALPHA and self.vars_A[neighbor_row][neighbor_col] >= 0:
                            a1 = self.vars_A[neighbor_row][neighbor_col]
                            V = self.lam

                            if max(np.abs(self.left_image[row][col] - self.left_image[neighbor_row][neighbor_col]), \
                                np.abs(self.right_image[row][col - alpha] - self.right_image[neighbor_row][neighbor_col - alpha])) < 8:
                                V = 3 * self.lam
                            
                            g.add_tedge(a1, 0, V)

                        # Case 3: Nonalpha disparity neighbors: both active
                        if self.vars_0[row][col] >= 0 and self.vars_0[neighbor_row][neighbor_col] >= 0 and \
                        self.d_left[row][col] == self.d_left[neighbor_row][neighbor_col]:
                            o1 = self.vars_0[row][col]
                            o2 = self.vars_0[neighbor_row][neighbor_col]
                            V = self.lam
                            
                            if max(np.abs(self.left_image[row][col] - self.left_image[neighbor_row][neighbor_col]), \
                                np.abs(self.right_image[row][col - self.d_left[row][col]] - self.right_image[neighbor_row][neighbor_col - self.d_left[row][col]])) < 8:
                                V = 3 * self.lam

                            self.add_term2(g, o1, o2, 0, V, V, 0)
                            # self.add_term2(g, o2, o1, 0, V, V, 0)

                        # Case 4: Nonalpha disparity neighbors: one active
                        if self.vars_0[row][col] >= 0 and not self.d_left[row][col] == self.d_left[neighbor_row][neighbor_col] and \
                        neighbor_col >= self.d_left[row][col]:
                            o1 = self.vars_0[row][col]
                            V = self.lam

                            if max(np.abs(self.left_image[row][col] - self.left_image[neighbor_row][neighbor_col]), \
                                np.abs(self.right_image[row][col - self.d_left[row][col]] - self.right_image[neighbor_row][neighbor_col - self.d_left[row][col]])) < 8:
                                V = 3 * self.lam

                            g.add_tedge(o1, 0, V)
                        elif self.vars_0[neighbor_row][neighbor_col] >= 0 and not self.d_left[row][col] == self.d_left[neighbor_row][neighbor_col] and \
                        col >= self.d_left[neighbor_row][neighbor_col]:
                            o1 = self.vars_0[neighbor_row][neighbor_col]
                            V = self.lam

                            if max(np.abs(self.left_image[row][col] - self.left_image[neighbor_row][neighbor_col]), \
                                np.abs(self.right_image[row][col - self.d_left[neighbor_row][neighbor_col]] - self.right_image[neighbor_row][neighbor_col - self.d_left[neighbor_row][neighbor_col]])) < 8:
                                V = 3 * self.lam

                            g.add_tedge(o1, 0, V)

    def create_unique_edges(self, g, alpha):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                # Enforce uniqueness in left image
                if self.vars_0[row][col] >= 0 and self.vars_A[row][col] >= 0:
                    o = self.vars_0[row][col]
                    a = self.vars_A[row][col]
                    self.add_term2(g, o, a, 0, 1e9, 0, 0)
                    # self.add_term2(g, a, o, 0, 1e9, 0, 0)

                # Enforce uniqueness in right image
                if self.vars_0[row][col + self.d_right[row][col]] >= 0 and col + alpha < self.num_cols and \
                self.vars_A[row][col + alpha] >= 0:
                    o = self.vars_0[row][col + self.d_right[row][col]]
                    a = self.vars_A[row][col + alpha]
                    self.add_term2(g, o, a, 0, 1e9, 0, 0)
                    # self.add_term2(g, a, o, 0, 1e9, 0, 0)

    def update_d_left_right(self, g, alpha):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if self.vars_0[row][col] >= 0 and g.get_segment(self.vars_0[row][col]) == 1:
                    disp = self.d_left[row][col]
                    self.d_left[row][col] = -1
                    self.d_right[row][col - disp] = -1
                if self.vars_A[row][col] >= 0 and g.get_segment(self.vars_A[row][col]) == 1:
                    self.d_left[row][col] = alpha
                    self.d_right[row][col - alpha] = alpha

    def alpha_expansion(self, alpha):
        self.reset_state()

        approx_num_vertices = self.num_rows * self.num_cols
        approx_num_edges = 4 * approx_num_vertices
        g = maxflow.Graph[int](approx_num_vertices, approx_num_edges)

        nodes = self.add_nodes(g, alpha)
        self.create_data_occlusion_edges(g, alpha)
        self.create_smoothness_edges(g, alpha)
        self.create_unique_edges(g, alpha)

        max_flow = g.maxflow()
        new_energy = max_flow + self.constant_additional_cut_cost

        if new_energy < self.curr_energy:
            self.curr_energy = new_energy
            print(new_energy)
            self.update_d_left_right(g, alpha)
            return True

        return False

    def expansion_move(self):
        random.shuffle(self.I_disp)

        for alpha in self.I_disp:
            if not self.done[alpha - 1]:
                improved = self.alpha_expansion(alpha)

                if improved:
                    self.done = [False for i in range(len(self.I_disp))]

                self.done[alpha - 1] = True

                if all(self.done):
                    return

    def minimize_energy(self):
        iterations = 0
        while not all(self.done):
            self.expansion_move()
            # iterations += 1
            # if iterations >= 65:
            #     break

        # Set disparity of occluded pixels to 0
        self.d_left[self.d_left == -1] = 0
        self.d_right[self.d_right == -1] = 0

        # flat_data = self.d_left.flatten()
        # plt.hist(flat_data, bins=np.arange(min(flat_data), max(flat_data) + 1) - 0.5, edgecolor='black')
        # plt.xlabel('Values')
        # plt.ylabel('Frequency')
        # plt.title('Histogram of 2D NumPy Array')
        # plt.show()

        # print(self.max_disparity)

        return self.d_left, self.d_right
