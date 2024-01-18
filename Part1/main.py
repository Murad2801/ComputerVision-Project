import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob


def isMatching(descriptors1, N, descriptors2, M, threshold):
    distances = np.sqrt(np.sum(np.square(descriptors1[:, np.newaxis] - descriptors2), axis=2))
    matchingPoints = []
    for i in range(N):
        sortedKeypointIndices = np.argsort(distances[i])
        smallest = distances[i][sortedKeypointIndices[0]]
        secondSmallest = distances[i][sortedKeypointIndices[1]]
        ratio = smallest / secondSmallest
        if ratio < threshold:
            matchingPoints.append((i, sortedKeypointIndices[0]))

    return matchingPoints


def solver(source, destination, puzzletype):
    src = np.float32(source)
    dst = np.float32(destination)
    if puzzletype == "affine":
        row_3 = np.array([0, 0, 1])
        row_3 = row_3.reshape((1, 3))
        M = np.append(cv2.getAffineTransform(src, dst), row_3, axis=0)
        return M
    if puzzleType == "homography":
        return cv2.getPerspectiveTransform(src, dst)


def transformation(M, src, num):
    homogeneous_points = np.hstack([src, [1]])
    transformed_points = np.matmul(homogeneous_points, M.T)
    res = transformed_points[:2] / transformed_points[2:]

    return res


def ransac(iterations, puzzletype, matching_src, matching_dst, ransac_threshold,num):
    best_inliers = None
    best_transformation_matrix = None

    for it in range(iterations):
        # Randomly select minimum number of points required for the transformation
        sample_indices = np.random.choice(len(matching_src), num, replace=False)
        sample_src = matching_src[sample_indices]
        sample_src = cv2.KeyPoint_convert(sample_src)
        sample_dst = matching_dst[sample_indices]
        sample_dst = cv2.KeyPoint_convert(sample_dst)

        # Calculate the transformation matrix
        transformation_matrix = solver(sample_src, sample_dst, puzzletype)
        new_matching_src = cv2.KeyPoint_convert(matching_src)
        new_matching_dst = cv2.KeyPoint_convert(matching_dst)
        residuals = []
        if 9 > np.linalg.det(transformation_matrix) > 0:
            for i in range(len(matching_src)):
                result = transformation(transformation_matrix, new_matching_src[i], num)
                result = np.array(result)

                residuals.append(np.linalg.norm((result - new_matching_dst[i])))

        #elif 0 > np.linalg.det(transformation_matrix) or np.linalg.det(transformation_matrix) > 9:
         #   continue
        residuals = np.array(residuals)

        # Calculate the inliers based on threshold
        inliers_ = new_matching_src[residuals < ransac_threshold]

        # Update the best transformation matrix if current iteration has more inliers than previous iterations
        if best_inliers is None or len(inliers_) > best_inliers:
            best_inliers = len(inliers_)
            best_transformation_matrix = transformation_matrix

    # Calculate the final transformation matrix using all the inliers.
    # transformation_matrix = transformation_func(best_inliers, dst_pts[best_inliers])

    return best_transformation_matrix, best_inliers


if __name__ == "__main__":
    #####################
    puzzleType = "homography"
    puzzleName = "puzzle_homography_10"
    height = 506
    width = 759

    background = np.zeros((height,width))
    #####################
    # read images
    imgArr = []
    files = glob.glob("./puzzles/" + puzzleName + "/pieces/*.jpg")
    for myFile in files:
        image = cv2.imread(myFile)
        imgArr.append(image)
    imgArrOriginal = imgArr.copy()
    if puzzleType == "affine":
        num = 3
    else:
        num = 4
    background_ones = np.ones((imgArr[0].shape[0],imgArr[0].shape[1]))
    warping_mat = np.loadtxt('./puzzles/'+puzzleName+'/warp_mat_1__H_'+str(height)+'__W_'+str(width)+'_.txt')
    imgArr[0] = cv2.warpPerspective(imgArr[0], warping_mat, (width, height), flags=2, borderMode=cv2.BORDER_TRANSPARENT)
    warpedcover1 = cv2.warpPerspective(background_ones, warping_mat, (width, height), flags=2, borderMode=cv2.BORDER_TRANSPARENT)
    cv2.imwrite('./results/' + puzzleName + '/piece_1_relative.jpeg',imgArr[0])
    background += warpedcover1
    # sift
    sift = cv2.SIFT_create()
    keyPoints = []
    descriptors = []
    for image in imgArr:
        temp1, temp2 = sift.detectAndCompute(image, None)
        keyPoints.append(temp1)
        descriptors.append(temp2)

    pieceNum = len(imgArr)
    counterr=0
    for piece in range(pieceNum-1):
        best_image_index = 0
        best_t_matrix = None
        best_image_inliers = 0

        # find best image :
        for j in range(1, len(imgArr)):
            flag = 0
            matchingPoints = isMatching(descriptors[0], len(keyPoints[0]), descriptors[j], len(keyPoints[j]), 0.85)
            matchingPoints = np.array(matchingPoints)
            if len(matchingPoints) >= num and best_image_inliers < len(matchingPoints):
                src_indices = matchingPoints[:, 0]
                src_kp = [keyPoints[0][h] for h in src_indices]
                src_kp = np.array(src_kp)
                dst_indices = matchingPoints[:, 1]
                dst_kp = [keyPoints[j][h] for h in dst_indices]
                dst_kp = np.array(dst_kp)
                t_matrix, inliers = ransac(10000, puzzleType, dst_kp, src_kp,  1, num)
                if best_image_inliers < inliers:
                    best_image_index = j
                    best_t_matrix = t_matrix
                    best_image_inliers = inliers

        #h, w = imgArr[0].shape[:2]
        index=0
        for i, img1 in enumerate(imgArrOriginal):
            if np.array_equal(img1, imgArr[best_image_index]):
                index = i + 1
                break

        best_t_matrix=np.array(best_t_matrix)
        if best_t_matrix.all() is not None:
            warped_img = cv2.warpPerspective(imgArr[best_image_index], best_t_matrix, (width, height), flags=2, borderMode=cv2.BORDER_TRANSPARENT)
            warpedcover = cv2.warpPerspective(background_ones, best_t_matrix, (width, height), flags=2, borderMode=cv2.BORDER_TRANSPARENT)
            background += warpedcover
            # stitching th pictures together
            mask_warped = (warped_img == 0)
            imgArr[0][~mask_warped] = warped_img[~mask_warped]
        else:
            print("piece number :" + str(index) + " not Matched!")
            continue

        keyPoints[0], descriptors[0] = sift.detectAndCompute(imgArr[0], None)
        keyPoints.pop(best_image_index)
        descriptors.pop(best_image_index)
        imgArr.pop(best_image_index)
        cv2.imwrite('./results/' + puzzleName + '/piece_' + str(index) + '_relative.jpeg', warped_img)
        counterr += 1
        cv2.imwrite('./results/' + puzzleName + '/solutions/solution_' + str(counterr) + '_' + str(pieceNum) + '.jpeg',
                    imgArr[0])
        plt.figure()
        plt.imshow(background)
        plt.savefig('./results/' + puzzleName + '/solutions/coverage count_' + str(counterr) + '.jpeg')

    cv2.imwrite('./results/' + puzzleName + '/solution_' + str(pieceNum) + '_' + str(pieceNum) + '.jpeg', imgArr[0])
    plt.figure()
    plt.imshow(background)
    plt.savefig('./results/' + puzzleName + '/coverage count.jpeg')



    # Display the result
    cv2.imshow('Blended Image', imgArr[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
