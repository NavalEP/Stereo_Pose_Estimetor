import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load stereo images (left and right)
left_img = cv2.imread(r"C:\Users\naval\OneDrive\Desktop\all\data\chess1\im0.png", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(r"C:\Users\naval\OneDrive\Desktop\all\data\chess1\im1.png", cv2.IMREAD_GRAYSCALE)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
keypoints_left, descriptors_left = orb.detectAndCompute(left_img, None)
keypoints_right, descriptors_right = orb.detectAndCompute(right_img, None)

# Initialize the BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Match descriptors
matches = bf.knnMatch(descriptors_left, descriptors_right, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw matches
matched_img = cv2.drawMatches(left_img, keypoints_left, right_img, keypoints_right, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# # Get the dimensions of your screen
# cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Matches', 1600, 900)
# Resize the image to fit the screen
screen_width, screen_height = 1600, 900  # Example screen resolution, adjust accordingly
height, width, _ = matched_img.shape

# Calculate the scaling factor to fit the screen
scaling_factor = min(screen_width / width, screen_height / height)
new_size = (int(width * scaling_factor), int(height * scaling_factor))
resized_img = cv2.resize(matched_img, new_size)



cv2.imshow('Matches', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Extract location of good matches
pts_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Load camera parameters (Intrinsic parameters)
# Assuming camera intrinsic parameters are known
fx = 1758.23  # Focal length in x-direction (in pixels)
fy = 1758.23  # Focal length in y-direction (in pixels)
cx = 953.34   # Principal point x-coordinate (in pixels)
cy = 552.29   # Principal point y-coordinate (in pixels)<p></p>

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# Find the essential matrix
E, mask = cv2.findEssentialMat(pts_left, pts_right, K)

# Recover the pose
_, R, t, mask = cv2.recoverPose(E, pts_left, pts_right, K)

# Print estimated rotation and translation
print("Rotation Matrix:\n", R)
print("Translation Vector:\n", t)


from scipy.spatial.transform import Rotation as R

def rotation_error(R_gt, R_est):
    # Ensure compatibility with scipy version
    r_gt = R.from_matrix(R_gt)
    r_est = R.from_matrix(R_est)
    relative_rotation = r_gt.inv() * r_est
    angle_error = relative_rotation.magnitude() * (180 / np.pi)
    return angle_error

# Assuming R_gt and t_gt are provided as numpy arrays
R_gt = np.array([[0.9998483, 0.0174524, 0],
                 [-0.0174524, 0.9998483, 0],
                 [0, 0, 1]])  # Example ground truth rotation matrix
t_gt = np.array([[0.1], [0], [0.95]])  # Example ground truth translation vector

# Estimated rotation and translation
R_est = np.array([[ 0.98266803,  0.00530552, -0.1852981 ],
                  [-0.00489294,  0.99998443,  0.00268381],
                  [ 0.18530946, -0.00173064,  0.98267869]])
t_est = np.array([[-0.1807296 ],
                  [-0.02449816],
                  [ 0.98322767]])

# Calculate rotation error
rotation_err = rotation_error(R_gt, R_est)
print("Rotation Error (degrees):", rotation_err)

# Calculate translation error
def translation_error(t_gt, t_est):
    return np.linalg.norm(t_gt - t_est)

translation_err = translation_error(t_gt, t_est)
print("Translation Error (meters):", translation_err)

# Example 3D points in world coordinates and their corresponding 2D points in the image plane
pts_3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])  # Replace with actual 3D points
pts_2d = np.array([[100, 100], [200, 100], [100, 200], [200, 200]])  # Replace with actual 2D points

# Assuming K is known (camera intrinsic matrix)
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])  # Replace with actual intrinsic parameters

# Calculate reprojection error
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

# Visualize Error Metrics
# rotation_error_value = 1.2  # Hypothetical value
# translation_error_value = 0.5  # Hypothetical value
# reprojection_error_value = 2.3  # Hypothetical value

errors = ['Rotation Error (degrees)', 'Translation Error (meters)', 'Reprojection Error (pixels)']
values = [rotation_err , translation_err, reproj_err ]

plt.figure(figsize=(10, 5))
plt.bar(errors, values, color=['blue', 'green', 'red'])
plt.title('Error Metrics')
plt.ylabel('Error Value')
plt.show()