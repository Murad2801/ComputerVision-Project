import numpy as np
import cv2


def census_transformation(image, height, width):
    vector_length = window_size*window_size-1
    census_matrix = np.zeros((height, width, vector_length), dtype=np.uint8)
    window_radius = window_size // 2
    for x in range(height):
        for y in range(width):
            center_pixel = image[x, y]
            vector = []
            for i in range(-window_radius, window_radius+1):
                for j in range(-window_radius, window_radius+1):
                    if i == 0 and j == 0:
                        continue
                    neighbor_x = x + i
                    neighbor_y = y + j
                    if 0 <= neighbor_x < height and 0 <= neighbor_y < width:
                        neighbor_pixel = image[neighbor_x, neighbor_y]
                        vector.append(1 if neighbor_pixel >= center_pixel else 0)
                    else:
                        vector.append(0)
            census_matrix[x, y] = vector
    return census_matrix


def calculate_cost_volume(mode, left_census, right_census, disparity, height, width):
    cost_volume_grid = np.zeros((height, width, disparity), dtype=np.float32)
    if mode == 'left':
        for disp in range(max_disparity):
            right_census_shifted = np.roll(right_census, disp, axis=1)
            xor = np.bitwise_xor(left_census, right_census_shifted)
            cost = np.sum(xor, axis=2)
            cost_volume_grid[..., disp] = cost.copy()

    elif mode == 'right':
        for disp in range(max_disparity):
            right_census_shifted = np.roll(left_census, -disp, axis=1)
            xor = np.bitwise_xor(right_census, right_census_shifted)
            cost = np.sum(xor, axis=2)
            cost_volume_grid[..., disp] = cost.copy()

    return cost_volume_grid


def perform_local_aggregation(cost_volume_grid, height, width, d):
    aggregated_cost_volume = np.zeros((height, width, d), dtype=np.float32)
    for disparity in range(d):
        cost_map = cost_volume_grid[:, :, disparity]
        aggregated_cost_map = cv2.blur(cost_map, (25, 25))
        aggregated_cost_volume[:, :, disparity] = aggregated_cost_map

    return aggregated_cost_volume


def consistency_test_filter(mode, first_disparity_map, second_disparity_map, height, width, threshold):
    consistency_map = np.zeros_like(first_disparity_map, dtype=np.float32)
    for x in range(height):
        for y in range(width):
            first_disparity = first_disparity_map[x, y]
            if mode == 'left':
                second_y = y - first_disparity
            else:  # mode == 'right'
                second_y = y + first_disparity
            # Check consistency
            if 0 <= second_y < width:
                second_disparity = second_disparity_map[x, second_y]
                if abs(first_disparity - second_disparity) <= threshold:
                    consistency_map[x, y] = first_disparity
                else:
                    consistency_map[x, y] = 0

    return consistency_map


def create_depth_map(K_matrix, disparity_map, height, width):
    focal_length = K_matrix[0][0]
    baseline = 0.1
    depth_map = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if disparity_map[i, j] > 0:
                depth_map[i, j] = (focal_length * baseline) / disparity_map[i, j]
            else:
                depth_map[i, j] = 0
    return depth_map


def reproject_2D_3D_2D(image, K_matrix, depth, P_matrix):

    canvas = np.zeros_like(image, dtype=np.uint8)
    height, width = image.shape[:2]
    K_inv = np.linalg.inv(K_matrix)
    # Converting the pixel coordinates [u,v] of the left image to homogeneous coordinates [u, v, 1]
    for u in range(height):
        for v in range(width):
            pixel_homogeneous = np.array([v, u, 1], dtype=np.float32)
            normalized_coords = np.matmul(K_inv, pixel_homogeneous)

            # Multiply normalized coordinates by depth value - converting to 3D point
            point_3d = depth[u, v] * normalized_coords

            # Convert back to 2D point
            x = np.append(point_3d, 1)
            point_2d = np.matmul(P_matrix, x.reshape(4, 1)).reshape(-1)

            # making sure to not divide by zero
            if point_2d[2] != 0:
                point_2d = point_2d / point_2d[2]

            # rounding the x,y coords and converting to int
            x = int(round(point_2d[1]))
            y = int(round(point_2d[0]))

            # making sure that the x , y coords are not out of bound
            if (x < 0 or x >= height) or (y < 0 or y >= width):
                continue

            canvas[x][y] = image[u][v]
    return canvas


def translation(left_depth, left_img, left_disp,  K_matrix, right_img):
    R_matrix = np.identity(3)
    T_vector = np.zeros((3, 1))
    X_matrix = np.concatenate((R_matrix, T_vector), axis=1)
    translate = 0.01
    baseline = 0.1
    focal_length = 1
    original_left_disp = left_disp.copy()
    added_value = (translate/baseline) * focal_length

    for i in range(11):
        translation_in_cm = i/100

        translation_matrix = np.identity(4)
        translation_matrix[0][3] = -translation_in_cm

        translated_X_matrix = np.matmul(X_matrix, translation_matrix)
        translated_P_matrix = np.matmul(K_matrix, translated_X_matrix)

        image = reproject_2D_3D_2D(left_img, K_matrix, left_depth, translated_P_matrix)

        ''' bonus for the silhouettes holes getting information from right image'''
        mask = (image==0) # mask for non-valid pixels in image
        image[mask]=right_img[mask]

        # cv2.imwrite('./sol_' + set_name + '/synth_' + str(i+1).zfill(2) + '.jpg', image)
        cv2.imwrite('./bonus_' + set_name + '/synth_' + str(i+1).zfill(2) + '.jpg', image)

        ''' bonus for the Thin vertical black lines '''
        nonzero_pixels_left_disp = (original_left_disp > 0)
        left_disp[nonzero_pixels_left_disp] = original_left_disp[nonzero_pixels_left_disp] + added_value
        left_depth = create_depth_map(K_matrix, left_disp, image.shape[0], image.shape[1])

def bonus_silhouettes_holes(left_disp, right_disp):

    # in-paint left disparity map using cv2
    silhouettes_mask = (left_disp < 0.5).astype(np.uint8)
    structuring_element  = np.ones((1, 1), dtype=np.uint8)
    expanded_mask = cv2.dilate(silhouettes_mask, structuring_element, iterations=1)
    smoothed_mask = cv2.erode(expanded_mask, structuring_element, iterations=1)
    left_disp_inpaint = cv2.inpaint(left_disp, smoothed_mask, 3, cv2.INPAINT_TELEA)

    # in-paint right disparity map using cv2
    silhouettes_mask = (right_disp < 0.5).astype(np.uint8)
    structuring_element  = np.ones((1, 1), dtype=np.uint8)
    expanded_mask = cv2.dilate(silhouettes_mask, structuring_element, iterations=1)
    smoothed_mask = cv2.erode(expanded_mask, structuring_element, iterations=1)
    right_disp_inpaint = cv2.inpaint(right_disp, smoothed_mask, 3, cv2.INPAINT_TELEA)

    # getting the new depth maps using the in-paint disparities
    left_depth_inpaint = create_depth_map(K_mat, cv2.blur(left_disp_inpaint, (5, 5)), im_height, im_width)
    right_depth_inpaint = create_depth_map(K_mat, cv2.blur(right_disp_inpaint, (5, 5)), im_height, im_width)

    # saving disparity photos in bonus file
    cv2.imwrite('./bonus_' + set_name + '/disp_left.jpg', cv2.normalize(left_disp_inpaint / max_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
    cv2.imwrite('./bonus_' + set_name + '/disp_right.jpg', cv2.normalize(right_disp_inpaint / max_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))

    # saving the depth photos in bonus file
    cv2.imwrite('./bonus_' + set_name + '/depth_left.jpg', cv2.normalize(left_depth_inpaint / max_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
    cv2.imwrite('./bonus_' + set_name + '/depth_right.jpg', cv2.normalize(right_depth_inpaint / max_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))

    return left_depth_inpaint, left_disp_inpaint



if __name__ == '__main__':
    window_size = 5
    set_name = "set_1"
    consistency_threshold = 0
    max_disparity = int(np.loadtxt("./" + set_name + "/max_disp.txt"))

    left_im = cv2.imread("./" + set_name + "/im_left.jpg")
    gray_left_im = cv2.cvtColor(left_im, cv2.COLOR_BGR2GRAY)

    right_im = cv2.imread("./" + set_name + "/im_right.jpg")
    gray_right_im = cv2.cvtColor(right_im, cv2.COLOR_BGR2GRAY)
    im_height, im_width = gray_left_im.shape[:2]

    # find census
    census_transform_left = census_transformation(gray_left_im, im_height, im_width)
    census_transform_right = census_transformation(gray_right_im, im_height, im_width)

    # calculate cost volume
    cost_volume_left = calculate_cost_volume('left', census_transform_left, census_transform_right, max_disparity, im_height, im_width)
    cost_volume_right = calculate_cost_volume('right', census_transform_left, census_transform_right, max_disparity, im_height, im_width)

    # perform aggregation
    left_aggregated = perform_local_aggregation(cost_volume_left, im_height, im_width, cost_volume_left.shape[2])
    right_aggregated = perform_local_aggregation(cost_volume_right, im_height, im_width, cost_volume_right.shape[2])

    # disparity map
    left_disp_map = np.argmin(left_aggregated, axis=2)
    right_disp_map = np.argmin(right_aggregated, axis=2)

    # filtered disparity map
    filtered_left_disp_map = consistency_test_filter('left', left_disp_map, right_disp_map, im_height, im_width, consistency_threshold)
    filtered_right_disp_map = consistency_test_filter('right', right_disp_map, left_disp_map, im_height, im_width, consistency_threshold)

    # load the K matrix
    K_mat = np.loadtxt("./" + set_name + "/K.txt")

    # create the depth maps
    left_depth_map = create_depth_map(K_mat, filtered_left_disp_map, im_height, im_width)
    right_depth_map = create_depth_map(K_mat, filtered_right_disp_map, im_height, im_width)

    # # saving disparity photos
    # cv2.imwrite('./sol_' + set_name + '/disp_left.jpg', cv2.normalize(filtered_left_disp_map / max_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
    # cv2.imwrite('./sol_' + set_name + '/disp_right.jpg', cv2.normalize(filtered_right_disp_map / max_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
    # # saving the depth photos
    # cv2.imwrite('./sol_' + set_name + '/depth_left.jpg', cv2.normalize(left_depth_map / max_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
    # cv2.imwrite('./sol_' + set_name + '/depth_right.jpg', cv2.normalize(right_depth_map / max_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))

    # bonus function for silhouettes_holes
    left_depth_map, filtered_left_disp_map = bonus_silhouettes_holes(filtered_left_disp_map, filtered_right_disp_map)

    translation(left_depth_map, left_im, filtered_left_disp_map, K_mat, right_im)

