import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time # Keep for potential timing if needed later
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor, resize


class StackedCNN(nn.Module):
    def __init__(self):
        super(StackedCNN, self).__init__()
        # Adjusted FC layer input based on assumed output size after pooling
        # Input: 1 channel, 64x64 image
        # conv1 -> relu -> pool: 32 channels, 32x32
        # conv2 -> relu -> pool: 64 channels, 16x16
        # conv3 -> relu -> pool: 128 channels, 8x8
        # Flattened size: 128 * 8 * 8 = 8192
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 128) # Corrected size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x shape: [batch_size, num_images, C, H, W] or [batch_size, C, H, W] for single image eval
        if x.dim() == 5: # Batched grid prediction
             batch_size, num_images, C, H, W = x.shape
             x = x.view(-1, C, H, W) # Reshape to [batch*num_images, C, H, W]
        elif x.dim() == 4: # Single image refinement prediction
             num_images = 1 # Effectively
             batch_size = x.shape[0]
             # Shape is already [batch_size, C, H, W]
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")

        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))

        # Make sure the flattened size matches fc1 input
        expected_flat_features = 128 * (x.shape[2] * x.shape[3]) # Calculate dynamically
        if expected_flat_features != 128 * 8 * 8:
             print(f"Warning: Feature map size after pooling is {x.shape[2]}x{x.shape[3]}, "
                   f"expected 8x8. Adjust fc1 layer input size if necessary.")
             # Update calculation if needed, though it should be 8x8 based on layers
             expected_flat_features = 128 * 8 * 8


        x = x.view(-1, expected_flat_features) # Flatten
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))

        if x.dim() == 5: # Reshape back for batched grid prediction output
             x = x.view(batch_size, num_images, -1)

        return x



def divide_into_grids(image: np.array, target_size=64) -> np.array:
    height, width = image.shape[:2]
    h_half, w_half = max(1, height // 2), max(1, width // 2)

    grids = []
    # Top-left, top-right, bottom-right, bottom-left
    grids.append(image[:h_half, :w_half])
    grids.append(image[:h_half, w_half:])
    grids.append(image[h_half:, w_half:])
    grids.append(image[h_half:, :w_half])
    # Center-left, center-top, center-right, center-bottom
    grids.append(image[h_half // 2 : h_half // 2 + h_half, :w_half])
    grids.append(image[:h_half, w_half // 2 : w_half // 2 + w_half])
    grids.append(image[h_half // 2 : h_half // 2 + h_half, w_half:])
    grids.append(image[h_half:, w_half // 2 : w_half // 2 + w_half])
    # Center
    grids.append(image[h_half // 2 : h_half + h_half // 2, w_half // 2 : w_half + w_half // 2])

    grids_resized = []
    for grid in grids:
        if grid.shape[0] > 0 and grid.shape[1] > 0:
             # Ensure input to resize is grayscale if model expects 1 channel
             if len(grid.shape) == 3:
                 grid_gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
             else:
                 grid_gray = grid
             resized = cv2.resize(grid_gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
             grids_resized.append(resized)
        else:
             # Append a blank image if grid is empty
             grids_resized.append(np.zeros((target_size, target_size), dtype=np.uint8))

    grids_resized = np.array(grids_resized, dtype=np.float32) / 255.0
    return grids_resized.reshape(9, 1, target_size, target_size) # Add channel dim

def divide_into_grids2(image: np.array, target_size=64) -> np.array:
    height, width = image.shape[:2]
    h_half, w_half = max(1, height // 2), max(1, width // 2)

    grids = []
    
    grids.append(image[:, :w_half])
    grids.append(image[:, w_half:])
    grids.append(image[:h_half, :])
    grids.append(image[h_half:, :])
    grids.append(image[h_half // 2 : h_half // 2 + h_half, :])
    grids.append(image[:, w_half // 2 : w_half // 2 + w_half])

    grids_resized = []
    for grid in grids:
        if grid.shape[0] > 0 and grid.shape[1] > 0:
            if len(grid.shape) == 3:
                grid_gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
            else:
                grid_gray = grid
            resized = cv2.resize(grid_gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
            grids_resized.append(resized)
        else:
             grids_resized.append(np.zeros((target_size, target_size), dtype=np.uint8))


    grids_resized = np.array(grids_resized, dtype=np.float32) / 255.0
    return grids_resized.reshape(6, 1, target_size, target_size) # Add channel dim



def find_coarse_bounding_box(prediction_array, image_shape, num_cell=9, threshold=0.5):
    """
    Calculates the bounding box
    Returns (xmin, ymin, xmax, ymax)
    """
    if prediction_array is None or len(prediction_array) == 0:
        return None

    good_bad_arr = (prediction_array > threshold).astype(int)
    if np.sum(good_bad_arr) ==0 : # No cells above threshold
        return None

    height, width = image_shape[:2] # Use shape passed as argument
    h_half, w_half = max(1, height // 2), max(1, width // 2)

    
    if num_cell == 9:
        # TL, TR, BR, BL, CL, CT, CR, CB, C
        cell_regions = [
            (0, 0, w_half, h_half), (w_half, 0, width, h_half),
            (w_half, h_half, width, height), (0, h_half, w_half, height),
            (0, h_half // 2, w_half, h_half // 2 + h_half),          # CL
            (w_half // 2, 0, w_half // 2 + w_half, h_half),          # CT
            (w_half, h_half // 2, width, h_half // 2 + h_half),      # CR
            (w_half // 2, h_half, w_half // 2 + w_half, height),      # CB
            (w_half // 2, h_half // 2, w_half // 2 + w_half, h_half // 2 + h_half) # C
        ]
    elif num_cell == 6:
        # L, R, T, B, CV, CH
         cell_regions = [
            (0, 0, w_half, height), (w_half, 0, width, height),
            (0, 0, width, h_half), (0, h_half, width, height),
            (0, h_half // 2, width, h_half // 2 + h_half),  # Center Vertical Stripe
            (w_half // 2, 0, w_half // 2 + w_half, height)  # Center Horizontal Stripe
        ]
    else:
        raise ValueError("num_cell must be 6 or 9.")

    good_indices = np.where(good_bad_arr == 1)[0]

    min_x_overall, min_y_overall = width, height
    max_x_overall, max_y_overall = 0, 0
    found_good_region = False

    for index in good_indices:
        if 0 <= index < len(cell_regions):
            x_start, y_start, x_end, y_end = cell_regions[index]
            min_x_overall = min(min_x_overall, x_start)
            min_y_overall = min(min_y_overall, y_start)
            max_x_overall = max(max_x_overall, x_end)
            max_y_overall = max(max_y_overall, y_end)
            found_good_region = True
        else:
            print(f"Warning: Index {index} out of bounds for cell_regions (size {len(cell_regions)})")


    if not found_good_region:
        return None

    # Ensure coordinates are within image bounds and valid (min < max)
    min_x_overall = max(0, min_x_overall)
    min_y_overall = max(0, min_y_overall)
    max_x_overall = min(width, max_x_overall)
    max_y_overall = min(height, max_y_overall)

    if max_x_overall <= min_x_overall or max_y_overall <= min_y_overall:
         return None # Invalid region

    return (min_x_overall, min_y_overall, max_x_overall, max_y_overall)







def sliding_window_refinementv2(cropped_image_gray, model, score_threshold, device="cpu"):
    """
    Finds an optimal bounding box within the grayscale cropped image using a sliding window approach.
    """
    if cropped_image_gray is None or cropped_image_gray.ndim != 2 or 0 in cropped_image_gray.shape:
        print("Warning: Invalid input image provided to refinement.")
        # Return an invalid state clearly distinguishable from a valid low score
        return ((0, 0, 0, 0), -float('inf'))

    H, W = cropped_image_gray.shape
    model_input_size = (64, 64) # Expected input size by the CNN

    model.to(device)
    model.eval()

    best_score = -float('inf')
    
    best_bbox = (0, 0, W, H)

    min_dim = min(H, W)
    scales = np.linspace(0.8, 1.0, 8) 
    window_sizes = [int(s * min_dim) for s in scales]
    window_sizes = [max(s, 16) for s in window_sizes if s > 0] 
    window_sizes = sorted(list(set(window_sizes)), reverse=True) 

    step_fraction = 0.15 

    found_better_box = False 

    with torch.no_grad(): 
        for size in window_sizes:
            
            win_h = min(size, H)
            win_w = min(size, W)

            if win_h < 16 or win_w < 16: # Skip if clamped size is too small
                 continue

            # Calculate step size (at least 1 pixel)
            step_h = max(1, int(win_h * step_fraction))
            step_w = max(1, int(win_w * step_fraction))

            # Iterate through possible top-left corners (y, x)
            for ymin in range(0, H - win_h + 1, step_h):
                for xmin in range(0, W - win_w + 1, step_w):
                    xmax = xmin + win_w
                    ymax = ymin + win_h

                    # Extract patch
                    patch = cropped_image_gray[ymin:ymax, xmin:xmax]

                    
                    img_tensor = to_tensor(patch) 

                    
                    img_resized = resize(img_tensor, model_input_size,
                                         interpolation=T.InterpolationMode.BILINEAR,
                                         antialias=True) 

                   
                    img_batch = img_resized.unsqueeze(0) #[1, 1, 64, 64]

                    
                    img_batch = img_batch.to(device)

                    
                    prediction_output, *_ = model(img_batch) 

                    
                    score = prediction_output.cpu().item()

                    
                    if score > best_score:
                        best_score = score
                        best_bbox = (xmin, ymin, xmax, ymax)
                        found_better_box = True # Mark that we found a window score

    
    if found_better_box and best_score >= 0.85*score_threshold:
        
        final_bbox = best_bbox
        final_score = best_score
    else:
        
        with torch.no_grad():
            
            full_img_tensor = to_tensor(cropped_image_gray) # [1, H, W]
            full_img_resized = resize(full_img_tensor, model_input_size,
                                      interpolation=T.InterpolationMode.BILINEAR,
                                      antialias=True) # [1, 64, 64]
            full_img_batch = full_img_resized.unsqueeze(0).to(device) # [1, 1, 64, 64]

            # Predict
            prediction_output, *_ = model(full_img_batch)
            full_score = prediction_output.cpu().item()
            # print(f"Full crop score: {full_score:.4f}")

            # Return the bounding box covering the entire input image and its score
            final_bbox = (0, 0, W, H)
            final_score = full_score

            

    return final_bbox, final_score
def sliding_window_refinementv3(cropped_image_gray, model, score_threshold, device="cpu"):
    """
    Finds the smallest bounding box within the grayscale cropped image whose score
    is above the threshold, using a sliding window approach. Version 3.
    """
    if cropped_image_gray is None or cropped_image_gray.ndim != 2 or 0 in cropped_image_gray.shape:
        print("Warning: Invalid input image provided to refinement.")
        
        return ((0, 0, 0, 0), -float('inf'))

    H, W = cropped_image_gray.shape
    model_input_size = (64, 64) 

    
    model.to(device)
    model.eval()

    found_bbox = None
    found_score = -float('inf')

   
    min_dim = min(H, W)
    
    scales = np.linspace(0.7, 0.9, 8) # Example: 8 scales
    window_sizes = [int(s * min_dim) for s in scales]
    window_sizes = [max(s, 16) for s in window_sizes if s > 0] # Min size 16x16, ensure > 0
    # Sort smallest first
    window_sizes = sorted(list(set(window_sizes))) 

    step_fraction = 0.15 

    
    stop_search = False
    with torch.no_grad(): 
        for size in window_sizes:
            if stop_search: break 

            
            win_h = min(size, H)
            win_w = min(size, W)

            if win_h < 16 or win_w < 16: # Skip if clamped size is too small
                continue

            # Calculate step size (at least 1 pixel)
            step_h = max(1, int(win_h * step_fraction))
            step_w = max(1, int(win_w * step_fraction))

            # Iterate through possible top-left corners (y, x)
            for ymin in range(0, H - win_h + 1, step_h):
                if stop_search: break # Exit if we found a suitable box in this size loop

                for xmin in range(0, W - win_w + 1, step_w):
                    xmax = xmin + win_w
                    ymax = ymin + win_h

                    
                    patch = cropped_image_gray[ymin:ymax, xmin:xmax]

                    
                    img_tensor = to_tensor(patch) # Shape: [1, win_h, win_w]
                    img_resized = resize(img_tensor, model_input_size,
                                         interpolation=T.InterpolationMode.BILINEAR,
                                         antialias=True) # Shape: [1, 64, 64]
                    img_batch = img_resized.unsqueeze(0) # Shape: [1, 1, 64, 64]
                    img_batch = img_batch.to(device)

                    # --- Prediction ---
                    prediction_output, *_ = model(img_batch)
                    score = prediction_output.cpu().item()

                    
                    if score >= 0.85*score_threshold:
                        
                        found_bbox = (xmin, ymin, xmax, ymax)
                        found_score = score
                        
                        stop_search = True 
                        break 

    
    if found_bbox is not None:
        # A window meeting the threshold was found
        final_bbox = found_bbox
        final_score = found_score
    else:
        
        with torch.no_grad():
            # Preprocess the full cropped_image_gray
            full_img_tensor = to_tensor(cropped_image_gray) # [1, H, W]
            full_img_resized = resize(full_img_tensor, model_input_size,
                                      interpolation=T.InterpolationMode.BILINEAR,
                                      antialias=True) # [1, 64, 64]
            full_img_batch = full_img_resized.unsqueeze(0).to(device) # [1, 1, 64, 64]

            # Predict
            prediction_output, *_ = model(full_img_batch)
            full_score = prediction_output.cpu().item()
            # print(f"Full crop score: {full_score:.4f}")

            # Return the bounding box covering the entire input image and its score
            final_bbox = (0, 0, W, H)
            final_score = full_score

    return final_bbox, final_score
def map_coords_to_original(relative_bbox, offset_coords):
    """
    Maps bounding box coordinates from a cropped region to the original image.

    Args:
        relative_bbox (tuple): (xmin, ymin, xmax, ymax) relative to the cropped region.
        offset_coords (tuple): (x_offset, y_offset) of the cropped region's top-left
                               corner within the original image.

    Returns:
        tuple: (xmin, ymin, xmax, ymax) in the original image's coordinate system.
    """
    if relative_bbox is None or offset_coords is None:
        return None

    rel_x_min, rel_y_min, rel_x_max, rel_y_max = relative_bbox
    off_x, off_y = offset_coords

    orig_x_min = rel_x_min + off_x
    orig_y_min = rel_y_min + off_y
    orig_x_max = rel_x_max + off_x
    orig_y_max = rel_y_max + off_y

    return (orig_x_min, orig_y_min, orig_x_max, orig_y_max)





def find_object_location_iterative(
    original_image, model, device='cpu',
    num_cell=9, threshold=0.5,
    min_region_ratio=0.1, # <-- CHANGED: Ratio threshold (e.g., 0.1 for 10%)
    max_iterations=5):
    
    if not (0 < min_region_ratio <= 1.0):
        print(f"Warning: min_region_ratio ({min_region_ratio}) should be between 0 and 1. Clamping to 0.1.")
        min_region_ratio = 0.1

    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        current_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    elif len(original_image.shape) == 2:
        current_image_gray = original_image.copy()
    else:
        raise ValueError("Input image must be grayscale or BGR")

    current_image_for_pred = original_image.copy()
    total_offset = (0, 0)
    original_shape = current_image_gray.shape[:2]
    h_orig, w_orig = original_shape # Store original dimensions

    print("Starting iterative search...")
    for i in range(max_iterations):
        print(f"Iteration {i+1}/{max_iterations}")
        h, w = current_image_gray.shape[:2]
        print(f"  Current region size: {w}x{h}, Offset: {total_offset}")

        
        width_threshold = w_orig * min_region_ratio
        height_threshold = h_orig * min_region_ratio
        is_width_small_enough = (w <= width_threshold)
        is_height_small_enough = (h <= height_threshold)

        if is_width_small_enough or is_height_small_enough:
            reasons = []
            if is_width_small_enough:
                reasons.append(f"width {w} <= {width_threshold:.1f} ({min_region_ratio*100:.0f}%)")
            if is_height_small_enough:
                reasons.append(f"height {h} <= {height_threshold:.1f} ({min_region_ratio*100:.0f}%)")
            print(f"  Region ratio threshold met ({', '.join(reasons)}). Moving to refinement.")
            break
        
        predictions,grid_stack = predict_image_grid2(model, current_image_for_pred, num_cell, device)
        """
        plot_sorted_grid_with_region(
            grid_stack,
            predictions,
            current_image_for_pred.copy(), # Plot the image *before* this iteration's crop
            num_cell,
            title=f"V1 Iteration {i+1} - Grids & Processed Region"
        )
        """
        
        
        
        # Find the coarse bounding box within the current region
        coarse_bbox_relative = find_coarse_bounding_box(predictions, current_image_gray.shape, num_cell, threshold)

        if coarse_bbox_relative is None:
            print("  No promising cells found in this iteration. Stopping.")
            #if i == 0: return None
            break # Use previous region for refinement

        x_min_rel, y_min_rel, x_max_rel, y_max_rel = coarse_bbox_relative
        if x_max_rel <= x_min_rel or y_max_rel <= y_min_rel:
             print("  Warning: Invalid coarse bounding box found. Stopping.")
             if i == 0: return None
             break

        # Update total offset
        total_offset = (total_offset[0] + x_min_rel, total_offset[1] + y_min_rel)

        # Crop the *grayscale* image for the next iteration's size check / refinement
        current_image_gray = current_image_gray[y_min_rel:y_max_rel, x_min_rel:x_max_rel]
        # Crop the *original type* image for the next prediction step
        current_image_for_pred = current_image_for_pred[y_min_rel:y_max_rel, x_min_rel:x_max_rel]

        if current_image_gray.size == 0:
             print("  Error: Cropped image is empty. Stopping.")
             return None

    else: # No break occurred
        print(f"  Max iterations ({max_iterations}) reached. Moving to refinement.")

    # --- Refinement Step (Unchanged logic) ---
    print(f"\nStarting refinement on final region (size {current_image_gray.shape[1]}x{current_image_gray.shape[0]})...")
    if current_image_gray.size == 0:
         print("  Cannot refine empty image.")
         return None
    
    dynamic_threshold = -float('inf') # Default value if calculation fails
    try:
        # Resize the entire current region to model input size
        resized_full_region = cv2.resize(current_image_gray, (64, 64), interpolation=cv2.INTER_AREA)
        # [batch=1, channel=1, H=64, W=64]
        full_region_tensor = torch.from_numpy(resized_full_region).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
        # Get prediction score
        with torch.no_grad():
            model.eval()
            full_region_prediction = model(full_region_tensor)
        dynamic_threshold = full_region_prediction.item()
        if dynamic_threshold<=0.6:#hereee
            return None
        print(f"  Dynamic refinement threshold (score of full region): {dynamic_threshold:.4f}")
    except Exception as e:
         print(f"  Warning: Could not calculate dynamic threshold for refinement: {e}. Using -inf.")
         
    
    refined_bbox_relative, refinement_score = sliding_window_refinementv3(
        current_image_gray,
        model,
        dynamic_threshold,
        device 
    )
    if refined_bbox_relative is None:
        print("  Refinement failed to find a bounding box.")
        final_coarse_bbox_relative = (0, 0, current_image_gray.shape[1], current_image_gray.shape[0])
        print("  Falling back to the final coarse region.")
        final_bbox_original = map_coords_to_original(final_coarse_bbox_relative, total_offset)
    else:
        print(f"  Refinement successful (Score: {refinement_score:.4f}). Mapping coordinates.")
        final_bbox_original = map_coords_to_original(refined_bbox_relative, total_offset)

    
    if final_bbox_original:
        x1, y1, x2, y2 = final_bbox_original
        h_orig_check, w_orig_check = original_shape 
        x1 = max(0, min(int(round(x1)), w_orig_check-1))
        y1 = max(0, min(int(round(y1)), h_orig_check-1))
        x2 = max(x1+1, min(int(round(x2)), w_orig_check)) 
        y2 = max(y1+1, min(int(round(y2)), h_orig_check)) 
        
        if x1 >= x2 or y1 >= y2:
             print("Warning: Final bounding box became invalid after clipping/rounding.")
             return None 
        final_bbox_original = (x1, y1, x2, y2)
        print(f"Final BBox (Original Coords): {final_bbox_original}")
    else:
         print("Final bbox calculation resulted in None.")

    return final_bbox_original



def plot_sorted_grid_with_region(grid_stack, predictions, current_region_image, num_cell, title="Sorted Grids & Current Region"):
    
    if grid_stack is None or predictions is None or len(predictions) != num_cell:
        print("Warning: Invalid input for plotting.")
        return

    # Ensure current_region_image is valid
    if current_region_image is None or current_region_image.size == 0:
        print("Warning: Cannot plot invalid current_region_image.")
        return

    sorted_indices = np.argsort(predictions)[::-1]
    # Ensure grid_stack has the expected shape [num_cell, channels, H, W]
    if grid_stack.shape[0] != num_cell:
        print(f"Warning: grid_stack shape mismatch {grid_stack.shape} vs expected num_cell {num_cell}")
        return
    sorted_grid = grid_stack[sorted_indices]
    sorted_probs = predictions[sorted_indices]

    
    if num_cell == 9:
        nrows, ncols = 3, 4 
    elif num_cell == 6:
        nrows, ncols = 2, 4 
    else:
        print(f"Warning: Plotting layout not defined for num_cell={num_cell}")
        # Fallback layout or return
        nrows, ncols = (num_cell + 1 + 3) // 4, 4 # Generic layout attempt

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3)) # Adjust figsize
    fig.suptitle(title, fontsize=12)

    axes_flat = axes.flat # Flatten the axes array for easy iteration
    for idx in range(nrows * ncols):
        ax = axes_flat[idx]
        if idx < num_cell: # Plot sorted grids
            # Make sure we access the correct channel (usually 0 for grayscale)
            ax.imshow(sorted_grid[idx, 0, :, :], cmap='gray', vmin=0, vmax=1) # Add vmin/vmax
            original_grid_index = sorted_indices[idx] # Get original index
            ax.set_title(f"Grid Idx {original_grid_index}\nProb: {sorted_probs[idx]:.3f}")
            ax.axis('off')
        elif idx == num_cell: # Plot current region in the next slot
             # Handle color vs grayscale for display
             if len(current_region_image.shape) == 3:
                 display_img = cv2.cvtColor(current_region_image, cv2.COLOR_BGR2RGB)
                 cmap_val = None
             else:
                 display_img = current_region_image
                 cmap_val = 'gray'
             ax.imshow(display_img, cmap=cmap_val)
             ax.set_title(f"Current Region\n({current_region_image.shape[1]}x{current_region_image.shape[0]})")
             ax.axis('off')
        else: # Turn off remaining axes
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.94]) # Adjust layout rect
    # Show plot immediately for debugging (blocks execution until closed)
    plt.show()

def predict_image_grid2(model, image, num_cell=9, device='cpu'):
    """Predicts probabilities for grids and returns probs and the grid stack."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        image_gray = image
    else: raise ValueError("Input image must be grayscale or BGR")

    # Generate grids
    if num_cell == 9: grid_stack = divide_into_grids(image_gray)
    elif num_cell == 6: grid_stack = divide_into_grids2(image_gray)
    else: raise ValueError("num_cell must be 6 or 9")

    # Prepare tensor
    grid_tensor = torch.from_numpy(grid_stack).unsqueeze(0).to(device) # Add batch dim

    # Predict
    with torch.no_grad():
        model.eval()
        predictions_tensor = model(grid_tensor) # Shape: [1, num_cell, 1]

    # Process results
    predictions = predictions_tensor.squeeze().cpu().numpy() # Shape: [num_cell]

    # Return both predictions and the generated grid stack
    return predictions, grid_stack



# --- Drawing and Display Functions ---

def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    """Draws a bounding box on the image."""
    if bbox is None:
        return image # Return unchanged image if no box

    img_copy = image.copy() # Avoid modifying original image
    xmin, ymin, xmax, ymax = [int(round(c)) for c in bbox] # Ensure integer coordinates

    # Check if coordinates are valid before drawing
    h, w = img_copy.shape[:2]
    if xmin < xmax and ymin < ymax and xmin >= 0 and ymin >= 0 and xmax <= w and ymax <= h:
         cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), color, thickness)
    else:
         print(f"Warning: Invalid coordinates for drawing: {bbox} on image size {w}x{h}")

    return img_copy

def display_results(original_image, final_bbox, title="Object Localization Result"):
    """Displays the original image with the final bounding box."""
    image_with_box = draw_bounding_box(original_image, final_bbox, color=(0, 0, 255), thickness=3)

    # Resize for display if needed
    # display_image = cv2.resize(image_with_box, (640, 480)) # Optional resize
    display_image = image_with_box

    cv2.imshow(title, cv2.resize(display_image,(640,480)))
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Plotting Functions (Unchanged, optional for debugging) ---
def plot_sorted_grid(grid_stack, predictions, title="Sorted Grid 9"):
    # Assuming grid_stack is [9, 1, H, W] and predictions is [9]
    if grid_stack is None or predictions is None or len(predictions) != 9: return
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_grid = grid_stack[sorted_indices]
    sorted_probs = predictions[sorted_indices]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(title, fontsize=11)
    for idx, ax in enumerate(axes.flat):
        if idx < 9:
            ax.imshow(sorted_grid[idx, 0, :, :], cmap='gray') # Index channel 0
            ax.set_title(f"Prob: {sorted_probs[idx]:.3f}")
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    IMAGE_PATH = "biden.jpg" # Replace with your image path
    MODEL_PATH = "final\BB3.pth"   # Replace with your model path
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CELLS = 9       # Or 6, depending on model/strategy
    PROB_THRESHOLD = 0.7 # Adjust based on model performance
    MIN_REGION_SIZE = 0.4 # Stop iteration when width or height is below this
    MAX_ITER = 10       # 5Max refinement iterations

    print(f"Using device: {DEVICE}")

    # --- Load Image ---
    # Load as color first for display, convert to gray inside functions when needed
    original_image = cv2.imread(IMAGE_PATH)
    if original_image is None:
        print(f"Error: Could not load image at {IMAGE_PATH}")
        exit()
    print(f"Loaded image: {IMAGE_PATH} (shape: {original_image.shape})")


    # --- Load Model ---
    try:
        model = StackedCNN()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        model.to(DEVICE)
        
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()


    # --- Run Localization ---
    start_time = time.time()
    final_bounding_box = find_object_location_iterative(
        original_image,
        model,
        device=DEVICE,
        num_cell=NUM_CELLS,
        threshold=PROB_THRESHOLD,
        min_region_ratio=MIN_REGION_SIZE,
        max_iterations=MAX_ITER
    )
    end_time = time.time()
    
    print(f"\nLocalization finished in {end_time - start_time:.2f} seconds.")

    # --- Display Results ---
    if final_bounding_box:
        print(f"\nObject found. Final BBox: {final_bounding_box}")
        display_results(original_image, final_bounding_box)
    else:
        print(final_bounding_box)
        print("\nObject could not be localized.")
        # Optionally display the original image anyway
        cv2.imshow("Object Not Found", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()