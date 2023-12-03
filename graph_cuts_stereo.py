import cv2
import maxflow
import numpy as np
import random

VAR_ALPHA = -1
VAR_ABSENT = -2
NEIGHBORS = [[0, 1], [1, 0]]

class GraphCutsStereo:
    def __init__(self, left_image, right_image):
        self.num_rows, self.num_cols = left_image.shape[:2]
        self.left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY).astype(int)
        self.right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY).astype(int)

        self.max_disparity = int(0.15 * self.num_cols)
        self.I_disp = list(range(self.max_disparity))
        self.done = [False for i in range(len(self.I_disp))]

        self.d_left = np.full((self.num_rows, self.num_cols), -1)
        self.d_right = np.full((self.num_rows, self.num_cols), -1)

        self.K = 30
        self.lambda_smooth = 10
        self.trim_cutoff = 30
        self.curr_energy = float('inf')
        self.constant_additional_cut_cost = 0
        self.vars_A = np.zeros((self.num_rows, self.num_cols), dtype=int)
        self.vars_0 = np.zeros((self.num_rows, self.num_cols), dtype=int)

    def reset_state(self):
        self.constant_additional_cut_cost = 0
        self.vars_A = np.zeros((self.num_rows, self.num_cols), dtype=int)
        self.vars_0 = np.zeros((self.num_rows, self.num_cols), dtype=int)

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

    def calculate_D_prime(self, I_p, I_q):
        return min(self.trim_cutoff, np.abs(I_p - I_q)) ** 2 - self.K

    def create_data_occlusion_edges(self, g, alpha):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                
                # Currently active assigment with disparity == alpha
                if self.vars_A[row][col] == VAR_ALPHA and self.vars_0[row][col] == VAR_ALPHA:
                    D_prime = self.calculate_D_prime(self.left_image[row][col], self.right_image[row][col - alpha])
                    self.constant_additional_cut_cost += D_prime

                # Currently inactive assigment with disparity == alpha
                if self.vars_A[row][col] >= 0:
                    a = self.vars_A[row][col]
                    D_prime = self.calculate_D_prime(self.left_image[row][col], self.right_image[row][col - alpha])
                    g.add_tedge(a, D_prime, 0)

                # Currently active assigment with disparity != alpha
                if self.vars_0[row][col] >= 0:
                    o = self.vars_0[row][col]
                    pixel_disparity = self.d_left[row][col]
                    D_prime = self.calculate_D_prime(self.left_image[row][col], self.right_image[row][col - pixel_disparity])
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
                            V = self.lambda_smooth
                            
                            if max(np.abs(self.left_image[row][col] - self.left_image[neighbor_row][neighbor_col]), \
                                np.abs(self.right_image[row][col - alpha] - self.right_image[neighbor_row][neighbor_col - alpha])) < 8:
                                V = 3 * self.lambda_smooth

                            self.add_term2(g, a1, a2, 0, V, V, 0)
                            # self.add_term2(g, a2, a1, 0, V, V, 0)

                        # Case 2: Alpha disparity neighbors: one inactive
                        if self.vars_A[row][col] >= 0 and self.vars_A[neighbor_row][neighbor_col] == VAR_ALPHA:
                            a1 = self.vars_A[row][col]
                            V = self.lambda_smooth

                            if max(np.abs(self.left_image[row][col] - self.left_image[neighbor_row][neighbor_col]), \
                                np.abs(self.right_image[row][col - alpha] - self.right_image[neighbor_row][neighbor_col - alpha])) < 8:
                                V = 3 * self.lambda_smooth

                            g.add_tedge(a1, 0, V)
                        elif self.vars_A[row][col] == VAR_ALPHA and self.vars_A[neighbor_row][neighbor_col] >= 0:
                            a1 = self.vars_A[neighbor_row][neighbor_col]
                            V = self.lambda_smooth

                            if max(np.abs(self.left_image[row][col] - self.left_image[neighbor_row][neighbor_col]), \
                                np.abs(self.right_image[row][col - alpha] - self.right_image[neighbor_row][neighbor_col - alpha])) < 8:
                                V = 3 * self.lambda_smooth
                            
                            g.add_tedge(a1, 0, V)

                        # Case 3: Nonalpha disparity neighbors: both active
                        if self.vars_0[row][col] >= 0 and self.vars_0[neighbor_row][neighbor_col] >= 0 and \
                        self.d_left[row][col] == self.d_left[neighbor_row][neighbor_col]:
                            o1 = self.vars_0[row][col]
                            o2 = self.vars_0[neighbor_row][neighbor_col]
                            V = self.lambda_smooth
                            disp = self.d_left[row][col]
                            
                            if max(np.abs(self.left_image[row][col] - self.left_image[neighbor_row][neighbor_col]), \
                                np.abs(self.right_image[row][col - disp] - self.right_image[neighbor_row][neighbor_col - disp])) < 8:
                                V = 3 * self.lambda_smooth

                            self.add_term2(g, o1, o2, 0, V, V, 0)
                            # self.add_term2(g, o2, o1, 0, V, V, 0)

                        # Case 4: Nonalpha disparity neighbors: one active
                        if self.vars_0[row][col] >= 0 and not self.d_left[row][col] == self.d_left[neighbor_row][neighbor_col] and \
                        neighbor_col >= self.d_left[row][col]:
                            o1 = self.vars_0[row][col]
                            V = self.lambda_smooth
                            disp = self.d_left[row][col]

                            if max(np.abs(self.left_image[row][col] - self.left_image[neighbor_row][neighbor_col]), \
                                np.abs(self.right_image[row][col - disp] - self.right_image[neighbor_row][neighbor_col - disp])) < 8:
                                V = 3 * self.lambda_smooth

                            g.add_tedge(o1, 0, V)
                        elif self.vars_0[neighbor_row][neighbor_col] >= 0 and not self.d_left[row][col] == self.d_left[neighbor_row][neighbor_col] and \
                        col >= self.d_left[neighbor_row][neighbor_col]:
                            o1 = self.vars_0[neighbor_row][neighbor_col]
                            V = self.lambda_smooth
                            disp = self.d_left[neighbor_row][neighbor_col]

                            if max(np.abs(self.left_image[row][col] - self.left_image[neighbor_row][neighbor_col]), \
                                np.abs(self.right_image[row][col - disp] - self.right_image[neighbor_row][neighbor_col - disp])) < 8:
                                V = 3 * self.lambda_smooth

                            g.add_tedge(o1, 0, V)

    def create_unique_edges(self, g, alpha):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                # Enforce uniqueness in left image
                if self.vars_0[row][col] >= 0 and self.vars_A[row][col] >= 0:
                    o = self.vars_0[row][col]
                    a = self.vars_A[row][col]
                    self.add_term2(g, o, a, 0, 1e10, 0, 0)
                    # self.add_term2(g, a, o, 0, 1e10, 0, 0)

                # Enforce uniqueness in right image
                disp_right = self.d_right[row][col]
                if self.vars_0[row][col + disp_right] >= 0 and col + alpha < self.num_cols and \
                self.vars_A[row][col + alpha] >= 0:
                    o = self.vars_0[row][col + disp_right]
                    a = self.vars_A[row][col + alpha]
                    self.add_term2(g, o, a, 0, 1e10, 0, 0)
                    # self.add_term2(g, a, o, 0, 1e10, 0, 0)

    def update_d_left_right(self, g, alpha):
        for row in range(self.num_rows):
            for col in range(self.num_cols):

                # Deactivate assignments
                if self.vars_0[row][col] >= 0 and g.get_segment(self.vars_0[row][col]) == 1:
                    disp = self.d_left[row][col]
                    self.d_left[row][col] = -1
                    self.d_right[row][col - disp] = -1
                
                # Activate assignments
                if self.vars_A[row][col] >= 0 and g.get_segment(self.vars_A[row][col]) == 1:
                    self.d_left[row][col] = alpha
                    self.d_right[row][col - alpha] = alpha

    def alpha_expansion(self, alpha):
        self.reset_state()

        # Initialize graph
        approx_num_vertices = self.num_rows * self.num_cols
        approx_num_edges = 4 * approx_num_vertices
        g = maxflow.Graph[int](approx_num_vertices, approx_num_edges)

        # Create graph
        nodes = self.add_nodes(g, alpha)
        self.create_data_occlusion_edges(g, alpha)
        self.create_smoothness_edges(g, alpha)
        self.create_unique_edges(g, alpha)

        # Calculate new energy
        max_flow = g.maxflow()
        new_energy = max_flow + self.constant_additional_cut_cost

        # Update state if energy decreases
        if new_energy < self.curr_energy:
            self.curr_energy = new_energy
            print(new_energy)
            self.update_d_left_right(g, alpha)
            return True

        return False

    def expansion_move(self):
        random.shuffle(self.I_disp)

        for alpha in self.I_disp:
            if not self.done[alpha]:
                improved = self.alpha_expansion(alpha)

                if improved:
                    self.done = [False for i in range(len(self.I_disp))]

                self.done[alpha] = True

                if all(self.done):
                    return

    def minimize_energy(self):
        while not all(self.done):
            self.expansion_move()

        # Set disparity of occluded pixels to 0 before returning
        self.d_left[self.d_left == -1] = 0
        return self.d_left
