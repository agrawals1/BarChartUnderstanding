import cv2
import numpy as np
import pytesseract
from mmdet.apis import init_detector, inference_detector
import mmcv
import itertools
import scipy
from scipy.interpolate import CubicSpline, interp1d
from ChartDete.infer_line import do_instance, get_xrange, get_kp, interpolate

# Define relative paths
CKPT = "ChartDete/iter_3000.pth"
CONFIG = "ChartDete/lineformer_swin_t_config.py"
IMG_PATH = "ChartDete/pic_large.png"
DEVICE = "cpu"
model_lin = init_detector(CONFIG, CKPT, device=DEVICE)

def get_dataseries(img, annot=None, to_clean=False, post_proc=False, mask_kp_sample_interval=10, return_masks=False):
    """
    Extract data series from the chart image.
    """
    clean_img = img
    inst_masks = do_instance(model_lin, clean_img, score_thr=0.3)
    inst_masks = [line_mask.astype(np.uint8) * 255 for line_mask in inst_masks]
    
    pred_ds = []
    for line_mask in inst_masks:
        x_range = get_xrange(line_mask)
        line_ds = get_kp(line_mask, interval=mask_kp_sample_interval, x_range=x_range, get_num_lines=False, get_center=True)
        line_ds = interpolate(line_ds, inter_type='linear')
        pred_ds.append(line_ds)

    return pred_ds

def get_line_vals(img):
    """
    Calculate the y values corresponding to the given x coordinates.
    """
    line_dataseries = get_dataseries(img, to_clean=False)
    xlabel_bboxes = filtered_result_LR[4]
    sorted_idx_xlabel = np.argsort(xlabel_bboxes[:, 0])
    xlabel_bboxes = xlabel_bboxes[sorted_idx_xlabel]
    
    ylabel_bboxes = filtered_result_LR[5]
    sorted_idx_ylabel = np.argsort(ylabel_bboxes[:, 1])
    ylabel_bboxes = ylabel_bboxes[sorted_idx_ylabel]
    
    xlabels_texts = extract_text_from_bboxes(lower_right, xlabel_bboxes)
    least_ylabel = int(extract_text_from_bboxes(lower_right, ylabel_bboxes[0].reshape((1, 5)))[0])
    most_ylabel = int(extract_text_from_bboxes(lower_right, ylabel_bboxes[-1].reshape((1, 5)))[0])
    
    x_coords_int = [int(np.mean(bbox[[0, 2]])) for bbox in xlabel_bboxes]
    least_ylabel_y_coord = int(np.mean(ylabel_bboxes[0][[1, 3]]))
    most_ylabel_y_coord = int(np.mean(ylabel_bboxes[-1][[1, 3]]))
    ylabel_height_delta = most_ylabel_y_coord - least_ylabel_y_coord
    
    y_line_coords = []
    for x in x_coords_int:
        y_coords_for_x = []
        for line in line_dataseries:
            for point in line:
                if point['x'] == x:
                    y_coords_for_x.append(point['y'])
                    break
        if y_coords_for_x:
            y_line_coords.append(np.mean(y_coords_for_x))
        else:
            y_line_coords.append(None)
    
    y_line_values = [(300 / ylabel_height_delta) * (most_ylabel_y_coord - y_line) for y_line in y_line_coords]
    
    result_dict = {xlabel: y_value for xlabel, y_value in zip(xlabels_texts, y_line_values)}
    
    return result_dict

def filter_bounding_boxes(bbox_list, threshold):
    """
    Filters bounding boxes based on a confidence threshold.
    """
    filtered_list = []
    for bboxes in bbox_list:
        confidence_scores = bboxes[:, 4]
        mask = confidence_scores >= threshold
        filtered_bboxes = bboxes[mask]
        filtered_list.append(filtered_bboxes)
    
    return filtered_list

def extract_text_from_bboxes(image, bboxes):
    """
    Extracts text from bounding boxes in an image using OCR.
    """
    texts = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox[:4].astype(int)
        cropped_img = image[y_min-5:y_max+5, x_min-5:x_max+5]
        text = pytesseract.image_to_string(cropped_img, config='--psm 6')
        texts.append(text.strip())
    return texts

def get_axis_values(result, img, class_idx, label, sort_param):
    """
    Get axis values from bounding boxes.
    """
    bboxes = result[class_idx]
    sorted_idx = np.argsort(bboxes[:, sort_param])
    bboxes = bboxes[sorted_idx]
    texts = extract_text_from_bboxes(img, bboxes)
    if sort_param:
        texts = texts[::-1]
    print(f"{label} axis values: {texts}")
    return texts

# Specify the path to model config and checkpoint file
config_file = 'ChartDete/model/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py'
checkpoint_file = 'ChartDete/model/checkpoint.pth'

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Test a single image and show the results
result = inference_detector(model, IMG_PATH)
result_filtered = filter_bounding_boxes(result, 0.4)

ylabel_bbs = result_filtered[5]
ylabel_bbs_argmin = np.argsort(ylabel_bbs[:, 0])[0]
px = ylabel_bbs[ylabel_bbs_argmin][0]
px = px - px / 50

plot_area_bbs = result_filtered[2]
plot_area_bbs_argmax = np.argsort(plot_area_bbs[:, 1])[-1]
py = plot_area_bbs[plot_area_bbs_argmax][1]
py = py - py / 50

# Load the image using OpenCV
img = cv2.imread(IMG_PATH)

# Divide the image into four parts
height, width = img.shape[:2]
px, py = int(px), int(py)

# Extract the upper right part
upper_right = img[0:py, px:width]

# Extract the lower right part
lower_right = img[py:height, px:width]

# Save the upper right part
cv2.imwrite('ChartDete/upper_right.png', upper_right)

# Save the lower right part
cv2.imwrite('ChartDete/lower_right.png', lower_right)

print("Upper right and lower right parts have been saved.")

result_UR = inference_detector(model, 'ChartDete/upper_right.png')
result_LR = inference_detector(model, 'ChartDete/lower_right.png')
result_UR = filter_bounding_boxes(result_UR, 0.4)
result_LR = filter_bounding_boxes(result_LR, 0.4)

# Filter out bounding boxes in lower_right part that lie to the right of the plot area
plot_area_bbs_LR = result_LR[2]
plot_area_x_max = np.max(plot_area_bbs_LR[:, 2])
filtered_result_LR = [bboxes[bboxes[:, 0] <= plot_area_x_max] for bboxes in result_LR]

# Extract x-axis and y-axis values from upper_right and lower_right
x_axis_values_UR = get_axis_values(result_UR, upper_right, 4, 'x', 0)
y_axis_values_UR = get_axis_values(result_UR, upper_right, 5, 'y', 1)

x_axis_values_LR = get_axis_values(filtered_result_LR, lower_right, 4, 'x', 0)
y_axis_values_LR = get_axis_values(filtered_result_LR, lower_right, 5, 'y', 1)

# Extract pairs of (x-axis, y-axis) values corresponding to the bars
def get_bar_values(result, img, y_class_idx, x_texts):
    y_bboxes = result[y_class_idx]
    sorted_idx = np.argsort(y_bboxes[:, 0])
    y_bboxes = y_bboxes[sorted_idx]
    y_texts = extract_text_from_bboxes(img, y_bboxes)
    return list(zip(x_texts, y_texts))

bar_values_UR = get_bar_values(result_UR, upper_right, 14, x_axis_values_UR)
bar_values_LR = get_bar_values(filtered_result_LR, lower_right, 14, x_axis_values_LR)

print(f"Upper right bar values: {bar_values_UR}")
print(f"Lower right bar values: {bar_values_LR}")

# Save the visualization results
model.show_result(upper_right, result_UR, out_file='ChartDete/sample_result_UR_final.jpg')
model.show_result(lower_right, result_LR, out_file='ChartDete/sample_result_LR_final.jpg')

line_vals = get_line_vals(lower_right)
print(line_vals)
