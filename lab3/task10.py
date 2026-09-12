import cv2
import numpy as np

painting = cv2.imread("painting.jpg")

if painting is None:
    print("Error: painting.jpg not found!")
    exit()


frame = cv2.imread("frame.jpg")

if frame is None:
    print("Error: frame.jpg not found!")
    exit()

scale = 0.45

scaled_width = int(painting.shape[1] * scale)
scaled_height = int(painting.shape[0] * scale)

scaled_painting = cv2.resize(
    painting,
    (scaled_width, scaled_height),
    interpolation=cv2.INTER_AREA
)

cv2.imwrite("01_scaled_painting.jpg", scaled_painting)

print("Step 1: Linear scaling completed.")

angle = 0

tx = 150
ty = 100

center = (
    scaled_painting.shape[1] // 2,
    scaled_painting.shape[0] // 2
)

M_rigid = cv2.getRotationMatrix2D(center, angle, 1.0)

M_rigid[0, 2] += tx
M_rigid[1, 2] += ty

canvas = np.zeros_like(frame)

rigid_painting = cv2.warpAffine(
    scaled_painting,
    M_rigid,
    (frame.shape[1], frame.shape[0])
)

cv2.imwrite("02_rigid_transformation.jpg", rigid_painting)

print("Step 2: Rigid transformation completed.")

src_points = np.float32([
    [0, 0],                              
    [scaled_width, 0],                 
    [scaled_width, scaled_height],      
    [0, scaled_height]                  
])


dst_points = np.float32([
    [250, 120],     
    [750, 180],     
    [700, 550],    
    [220, 480]      
])

H = cv2.getPerspectiveTransform(
    src_points,
    dst_points
)

print("\nProjective Transformation Matrix:")
print(H)

warped_painting = cv2.warpPerspective(
    scaled_painting,
    H,
    (frame.shape[1], frame.shape[0])
)

cv2.imwrite(
    "03_projective_transformation.jpg",
    warped_painting
)

print("Step 3: Projective transformation completed.")


gray = cv2.cvtColor(warped_painting, cv2.COLOR_BGR2GRAY)

_, mask = cv2.threshold(
    gray,
    1,
    255,
    cv2.THRESH_BINARY
)

mask_inv = cv2.bitwise_not(mask)

background = cv2.bitwise_and(
    frame,
    frame,
    mask=mask_inv
)
foreground = cv2.bitwise_and(
    warped_painting,
    warped_painting,
    mask=mask
)

final_result = cv2.add(
    background,
    foreground
)

cv2.imwrite(
    "04_final_master_forger.jpg",
    final_result
)

cv2.imshow("Original Painting", painting)
cv2.imshow("Scaled Painting", scaled_painting)
cv2.imshow("Rigid Transformation", rigid_painting)
cv2.imshow("Projective Transformation", warped_painting)
cv2.imshow("Final Result", final_result)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nAll transformations completed successfully!")