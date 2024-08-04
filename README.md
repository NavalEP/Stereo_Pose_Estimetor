
    - # Stereo Vision: Estimating Camera Pose from Stereo Images

This project demonstrates the process of estimating the camera pose (rotation and translation) from a pair of stereo images using the ORB feature detector and descriptor, BFMatcher for feature matching, and calculating the essential matrix to recover the camera pose. Additionally, it includes error metrics for evaluating the accuracy of the estimated camera pose.

## Prerequisites

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Scipy

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. Install the required packages:
    ```bash
    pip install opencv-python numpy matplotlib scipy
    ```

## Usage

1. **Load Stereo Images:**
    ```python
    left_img = cv2.imread("path/to/left_image.png", cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread("path/to/right_image.png", cv2.IMREAD_GRAYSCALE)
    ```

2. **Initialize ORB Detector and Detect Keypoints:**
    ```python
    orb = cv2.ORB_create()
    keypoints_left, descriptors_left = orb.detectAndCompute(left_img, None)
    keypoints_right, descriptors_right = orb.detectAndCompute(right_img, None)
    ```

3. **Match Descriptors using BFMatcher and Apply Lowe's Ratio Test:**
    ```python
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors_left, descriptors_right, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    ```

4. **Draw Matches and Display:**
    ```python
    matched_img = cv2.drawMatches(left_img, keypoints_left, right_img, keypoints_right, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    screen_width, screen_height = 1600, 900  # Adjust accordingly
    height, width, _ = matched_img.shape
    scaling_factor = min(screen_width / width, screen_height / height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    resized_img = cv2.resize(matched_img, new_size)

    cv2.imshow('Matches', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

5. **Extract Locations of Good Matches:**
    ```python
    pts_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ```

6. **Load Camera Parameters and Find Essential Matrix:**
    ```python
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Define fx, fy, cx, cy accordingly
    E, mask = cv2.findEssentialMat(pts_left, pts_right, K)
    ```

7. **Recover the Pose:**
    ```python
    _, R, t, mask = cv2.recoverPose(E, pts_left, pts_right, K)
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", t)
    ```

8. **Calculate Error Metrics:**
    - **Rotation Error:**
        ```python
        from scipy.spatial.transform import Rotation as R

        def rotation_error(R_gt, R_est):
            r_gt = R.from_matrix(R_gt)
            r_est = R.from_matrix(R_est)
            relative_rotation = r_gt.inv() * r_est
            angle_error = relative_rotation.magnitude() * (180 / np.pi)
            return angle_error

        rotation_err = rotation_error(R_gt, R_est)
        print("Rotation Error (degrees):", rotation_err)
        ```

    - **Translation Error:**
        ```python
        def translation_error(t_gt, t_est):
            return np.linalg.norm(t_gt - t_est)

        translation_err = translation_error(t_gt, t_est)
        print("Translation Error (meters):", translation_err)
        ```

    - **Reprojection Error:**
        ```python
        def reprojection_error(pts_3d, pts_2d, R, t, K):
            pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))  # Convert to homogeneous coordinates
            if R.shape == (3, 3):
                R = np.hstack((R, np.zeros((3, 1))))  # Convert (3, 3) to (3, 4)
            elif R.shape != (3, 4):
                raise ValueError("Invalid shape for rotation matrix R: {}".format(R.shape))
            proj_points = K @ (R @ pts_3d_hom.T + t)
            proj_points /= proj_points[2]  # Normalize by the third (homogeneous) coordinate
            proj_points = proj_points[:2].T  # Convert back to Cartesian coordinates

            error = np.linalg.norm(pts_2d - proj_points, axis=1)
            return np.mean(error)

        reproj_err = reprojection_error(pts_3d, pts_2d, R_est, t_est, K)
        print("Reprojection Error (pixels):", reproj_err)
        ```

9. **Visualize Error Metrics:**
    ```python
    errors = ['Rotation Error (degrees)', 'Translation Error (meters)', 'Reprojection Error (pixels)']
    values = [rotation_err , translation_err, reproj_err ]

    plt.figure(figsize=(10, 5))
    plt.bar(errors, values, color=['blue', 'green', 'red'])
    plt.title('Error Metrics')
    plt.ylabel('Error Value')
    plt.show()
    ```

## Example Results

- **Rotation Matrix:**
    ```
     [[ 0.99683914 -0.00317034 -0.07938313],
     [ 0.00277546  0.99998322 -0.00508421],
     [ 0.07939792  0.00484782  0.99683121]]
    ```

- **Translation Vector:**
    ```
    [[-0.63701281],
    [ 0.07517822],
    [ 0.76717854]]
    ```

- **Error Metrics:**
    - Rotation Error (degrees): 10.702841765903555
    - Translation Error (meters): 0.2837487378177646
    - Reprojection Error (pixels): 1879.3703928479454

## Acknowledgments

- OpenCV
- NumPy
- Matplotlib
- Scipy




