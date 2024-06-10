import itertools
import scipy
from scipy.interpolate import CubicSpline, interp1d
from mmdet.apis import init_detector, inference_detector
import numpy as np

def parse_result(result, score_thresh=0.3):
    line_data = result
    bbox, masks = line_data[0][0], line_data[1][0]
    inst_masks = list(itertools.compress(masks, ((bbox[:, 4] > score_thresh).tolist())))
    return inst_masks


def do_instance(model, img, score_thr=0.3):
    # test a single image
    result = inference_detector(model, img)
    return parse_result(result, score_thr)

def get_kp(line_img, interval=10, x_range=None, get_num_lines=False, get_center=True):
    """
        line_img: np.ndarray => black and white binary mask of line
        black => background => 0
        white => foregrond line pixel => 255
        interval: delta_x at which x,y points are sampled across the line_img
        x_range: Range of x values, [xmin, xmax), within which pred points (x,y) are to be sampled
        returns: a list [{'x': <x_val>, 'y': <y_val>}, ....] of line points found in the binary line_img
    """

    im_h, im_w = line_img.shape[:2]
    kps = []
    if x_range is None:
        x_range = (0, im_w)

    # track the number of vertical binary components found at every x => estimate num lines
    num_comps = []
    for x in range(x_range[0], x_range[1], interval):
        # get the corresponding white pixel in this column
        fg_y = []
        fg_y_center = []
        all_y_points = np.where(line_img[:, x] == 255)[0]
        if all_y_points.size != 0:
            fg_y.append(all_y_points[0])
            y = all_y_points[0]
            n_comps = 1
            for idx in range(1, len(all_y_points)):
                y_next = all_y_points[idx]
                if abs(y_next - y) > 2:
                    n_comps += 1
                    # break found b/w y_next and y, separate components
                    if fg_y[-1] != y:
                        # handle the case where (first component itself is broken, i.e found break at idx=1)
                        fg_y_center.append(round(y + fg_y[-1])//2)
                        fg_y.append(y)
                    else:
                        fg_y_center.append(y)

                    fg_y.append(y_next)

                y = y_next
               
            if fg_y[-1] != y:
                # add the last point
                fg_y_center.append(round(y + fg_y[-1])//2)
                fg_y.append(y)
            else:
                fg_y_center.append(y)

            num_comps.append(n_comps)

        if (fg_y or fg_y_center) and (n_comps==1):
            if get_center:
                kps.extend([{'x':float(x), 'y':y} for y in fg_y_center])
            else:
                kps.extend([{'x':float(x), 'y':y} for y in fg_y])

    res = kps

    if get_num_lines:
        res = kps, int(np.percentile(num_comps, 85))

    return res



def get_xrange(bin_line_mask):
    """
        bin_line_mask: np.ndarray => black and white binary mask of line
        black => background => 0
        white => foregrond line pixel => 255
        returns: (x_start, x_end) where x_start and x_end represent the starting and ending points
                for the binary line segment
    """
    
    smooth_signal = scipy.signal.medfilt(bin_line_mask.sum(axis=0), kernel_size=5)
    x_range = np.nonzero(smooth_signal)
    if len(x_range) and len(x_range[0]): # To handle cases with empty masks
        x_range = x_range[0][[0, -1]]
    else:
        x_range = None
    return x_range

def interpolate(line_ds, inter_type='linear'):
    """
    pred_ds: predicted data series
    inter_type: type of interpolation linear or cubic_spline
    returns list of interpolation objects for each line in the mask

    """

    x = []
    y = []

    for pt in line_ds:
        x.append(pt['x'])
        y.append(pt['y'])

    # Remove duplicates
    unique_x = []
    unique_y = []

    for i in range(len(x)):
        if x.count(x[i]) == 1:
            unique_x.append(int(x[i]))
            unique_y.append(int(y[i]))

    if len(unique_x) < 2:
        return line_ds

    # Interpolate
    if inter_type == 'linear':
        inter = interp1d(unique_x, unique_y)
    if inter_type == 'cubic_spline':
        inter = CubicSpline(unique_x, unique_y)

    inter_line_ds = []
    x_min = min(unique_x)
    x_max = max(unique_x)

    for x in range(x_min, x_max+1):
        inter_line_ds.append({"x":x, "y":int(inter(x))})

    return inter_line_ds



