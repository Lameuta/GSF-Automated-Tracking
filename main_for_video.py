import cv2
import numpy as np
from better import StackedCNN, find_object_location_iterative
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time # Keep for potential timing if needed later
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor, resize

def convert_bbox_to_selectROI_format(bbox):
    """
    Converts a bounding box from (xmin, ymin, xmax, ymax) to (x, y, width, height) format.
    
    """
    if bbox is None:
        return None
    
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    
    return (xmin, ymin, width, height)

def compute_spatial_gradients(image):
    grad_x = np.gradient(image.astype(float), axis=1)  
    grad_y = np.gradient(image.astype(float), axis=0)  
    return grad_x, grad_y

def simple_tracker2(roi, template, im1, max_iterations=50, threshold=1e-3):
    # extract the initial details
    x, y, w, h = roi
    template = template.astype(np.float32) 
    im1 = im1.astype(np.float32)
    dx, dy = 0, 0
    
    for _ in range(max_iterations):
        # Warp the region in the next frame
        x_new, y_new = int(x + dx), int(y + dy)
        warped_image = im1[y_new:y_new + h, x_new:x_new + w]
        h1, w1 = min(template.shape[0], warped_image.shape[0]), min(template.shape[1], warped_image.shape[1])


        template = template[:h1, :w1]
        warped_image = warped_image[:h1, :w1]
        
        #  error image
        error = template - warped_image
        
        # image gradients
        grad_x, grad_y = compute_spatial_gradients(warped_image)

        # Flatten error and gradients 
        error_flat = error.flatten()
        grad_x_flat = grad_x.flatten()
        grad_y_flat = grad_y.flatten()
        
        A = np.vstack((grad_x_flat, grad_y_flat)).T
        b = error_flat

        if A.shape[0] != b.shape[0]:
            min_length = min(A.shape[0], b.shape[0])
            A = A[:min_length, :]
            b = b[:min_length]

        # solve for (delta_dx, delta_dy) using linear least squares
        delta_p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Update parameters
        dx += delta_p[0]
        dy += delta_p[1]

        if np.linalg.norm(delta_p) / np.linalg.norm([dx, dy]) < threshold:
            break
    
    # return updated ROI
    new_x, new_y = x + dx, y + dy
    return int(new_x), int(new_y), w, h


def pyramidal_tracker(roi, im0, im1, levels=3, scale=2, max_iterations=50, threshold=1e-3):
    
    # create Gaussian pyramid for both the image (im1) and the template
    pyramid_im1 = create_gaussian_pyramid(im1, levels)
    x, y, w, h = roi
    template = im0[y:y+h, x:x+w]
    pyramid_template = create_gaussian_pyramid(template, levels)
    #rescale the rois
    x, y, w, h = roi
    x /= scale**(levels-1)
    y /= scale**(levels-1)
    w /= scale**(levels-1)
    h /= scale**(levels-1)
    roi = int(round(x)), int(round(y)), int(round(w)), int(round(h))

    # start from the coarsest level 
    for level in range(levels - 1, -1, -1):
        # Get the current level's image and template
        current_im1 = pyramid_im1[level]
        current_template = pyramid_template[level]
        #cv2.imshow
        #upscale the rois
        if level != levels-1:
            roi = upscale_roi(roi, scale)
        
        
        # track the ROI at this level using simple SSD tracker
        roi = simple_tracker2(roi, current_template, current_im1, max_iterations, threshold)
        
    return roi, template


def create_gaussian_pyramid(image, levels):
    pyramid = [image]
    for i in range(1, levels):
        image = cv2.pyrDown(image)  #apply Gaussian downsampling 
        pyramid.append(image)
    return pyramid


def upscale_roi(roi, scale):
    x, y, w, h = roi
    x *= scale
    y *= scale
    w *= scale
    h *= scale
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))




if __name__ == "__main__":
    video_path = 'test_vid3.mp4'
    model = StackedCNN()
    model.load_state_dict(torch.load("final\BB3.pth", map_location=torch.device('cpu')))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    cam = cv2.VideoCapture(video_path)
    frame_count = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define codec
    out = cv2.VideoWriter('my_test_video_tracked_4.mp4', fourcc, 20.0, (640, 480))  # Output file
    template = None
    
    while True:
        ret, frame = cam.read()

    
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_count == 0:
            roi = convert_bbox_to_selectROI_format(find_object_location_iterative(gray_frame,model,device,9,0.7,0.4,10))
            im0 = gray_frame
            

        
        if frame_count !=0:
            if roi is not None:
                roi, template = pyramidal_tracker(roi, im0, gray_frame, levels=4, scale=2, max_iterations=25, threshold=0.0001)
                if roi is not None:
                    x, y, w, h = roi
            im0 = gray_frame#[y:y+h, x:x+w]
            if frame_count%10==0 or roi is None:
                print("update ROI")
                if template is not None:
                    resized_template = cv2.resize(template, (64, 64), interpolation=cv2.INTER_AREA)
                    template_tensor = torch.from_numpy(resized_template).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
            # Get prediction score for the full region
                    with torch.no_grad():
                        model.eval()
                        template_prediction = model(template_tensor)
                        dynamic_threshold = template_prediction.item()
                        if dynamic_threshold<=0.8:
                            print("update ROI")
                            roi = convert_bbox_to_selectROI_format(find_object_location_iterative(gray_frame,model,device,9,0.7,0.4,10))
                            if roi is not None:
                                roi, template = pyramidal_tracker(roi, im0, gray_frame, levels=4, scale=2, max_iterations=25, threshold=0.0001)
                else:
                    roi = convert_bbox_to_selectROI_format(find_object_location_iterative(gray_frame,model,device,9,0.7,0.4,10))
                    if roi is not None:
                        roi, template = pyramidal_tracker(roi, im0, gray_frame, levels=4, scale=2, max_iterations=25, threshold=0.0001)
                        if roi is not None:
                            x, y, w, h = roi
            
            
            result = frame.copy()
            if roi is not None:
                x, y, w, h = roi
                cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            result = cv2.resize(result,(640,480))
            cv2.imshow("Tracked ROI", result)
            out.write(result)

        k = cv2.waitKey(1)
        if k%256 == 27:
        # ASCII:ESC pressed
            print("Escape hit, closing...")
            break
        frame_count = frame_count+1


    cam.release()
    cv2.destroyAllWindows()