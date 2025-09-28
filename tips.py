import cv2
import os
import json
from glob import glob

mask_dir = "../model_input/mask_Catheter_Whole_RANZCR/validation"  
save_json = "catheter_tips_validation.json"

tip_dict = {}

mask_paths = sorted(glob(os.path.join(mask_dir, "*.jpg")))

for path in mask_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # resize to match training resolution
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    print(f"Click on the TIP for {os.path.basename(path)}. Press 's' to save, 'n' to skip, ESC to quit.")

    tip = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            tip.clear()
            tip.append((x, y))
            temp = disp.copy()
            cv2.circle(temp, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select Tip", temp)

    cv2.imshow("Select Tip", disp)
    cv2.setMouseCallback("Select Tip", mouse_callback)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("s") and tip:
            tip_dict[os.path.basename(path)] = tip[0]
            print(f"Saved tip {tip[0]} for {os.path.basename(path)}")
            break
        elif key == ord("n"):
            print("Skipped")
            break
        elif key == 27:  # ESC to quit early
            break

    cv2.destroyAllWindows()

# save to JSON
with open(save_json, "w") as f:
    json.dump(tip_dict, f, indent=2)
print(f"Saved {len(tip_dict)} tips to {save_json}")

def connected_component(inp, oup, dist_thresh=100):
    os.makedirs(oup, exist_ok=True)
    vis_dir = os.path.join(os.path.dirname(oup), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    mask_files = glob.glob(os.path.join(inp, "*_mask.jpg"))

    def get_endpoints(mask):
        """Find endpoints: pixels with exactly one neighbor in skeleton."""
        endpoints = []
        h, w = mask.shape
        for y in range(1, h-1):
            for x in range(1, w-1):
                if mask[y, x] == 1:
                    if mask[y-1:y+2, x-1:x+2].sum() == 2:  # self + one neighbor
                        endpoints.append((y, x))
        return endpoints

    for mask_file in mask_files:
        image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        _, binary = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8))
        component_sizes = [(np.sum(labels == l), l) for l in range(1, num_labels)]
        component_sizes.sort(key=lambda x: x[0], reverse=True)

        # If no components found, just save binary and skip
        if not component_sizes:
            cv2.imwrite(os.path.join(oup, os.path.basename(mask_file)), binary * 255)
            continue

        # Always keep largest component in final mask
        _, label1 = component_sizes[0]
        mask1 = (labels == label1).astype(np.uint8)

        # Initialize output mask as largest component
        final_mask = mask1.copy()

        if len(component_sizes) > 1:
            _, label2 = component_sizes[1]
            mask2 = (labels == label2).astype(np.uint8)

            # Skeletonize both components independently
            skel1 = skeletonize(mask1 > 0).astype(np.uint8)
            skel2 = skeletonize(mask2 > 0).astype(np.uint8)

            # Get endpoints independently
            endpoints1 = get_endpoints(skel1)
            endpoints2 = get_endpoints(skel2)

            # Default values if endpoints are missing
            closest_dist = 1e9
            closest_pair = None

            if endpoints1 and endpoints2:
                for pt1 in endpoints1:
                    for pt2 in endpoints2:
                        d = distance.euclidean(pt1, pt2)
                        if d < closest_dist:
                            closest_dist = d
                            closest_pair = (pt1, pt2)

            # Draw visualization
            vis = cv2.cvtColor((binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Mark endpoints from comp1 (red)
            for y, x in endpoints1:
                cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

            # Mark endpoints from comp2 (green)
            for y, x in endpoints2:
                cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)

            # Draw closest pair line in blue
            if closest_pair:
                (y1, x1), (y2, x2) = closest_pair
                cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

                print(f"{os.path.basename(mask_file)} - Closest tip distance: {closest_dist:.2f}")

                # Merge components if below distance threshold
                if closest_dist < dist_thresh:
                    final_mask = mask1 | mask2
            else:
                print(f"{os.path.basename(mask_file)} - No valid endpoints found.")

            # Save visualization
            vis_file = os.path.join(vis_dir, os.path.basename(mask_file).replace("_mask", "_vis.jpg"))
            cv2.imwrite(vis_file, vis)

        # Save the final binary mask
        cv2.imwrite(os.path.join(oup, os.path.basename(mask_file)), final_mask * 255)

    print("Connected component process complete.")