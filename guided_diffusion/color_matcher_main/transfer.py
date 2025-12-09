import cv2
import numpy as np

def color_transfer(source, reference):
    """
    Transfer color distribution from reference to source image using LAB mean/std.
    Both source and reference should be in BGR (OpenCV default).
    Returns a BGR image.
    """
    # Convert to LAB color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Compute mean and std dev
    src_mean, src_std = cv2.meanStdDev(source_lab)
    ref_mean, ref_std = cv2.meanStdDev(reference_lab)

    # Subtract mean, scale by std, add reference mean/std
    result_lab = (source_lab - src_mean.T) / (src_std.T + 1e-6)
    result_lab = result_lab * ref_std.T + ref_mean.T

    # Clip and convert back to BGR
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    return result_bgr


src = cv2.imread("/data/wenjieli/code/BFSR/PGDiff/codeformer.png")    
ref = cv2.imread("/data/wenjieli/code/BFSR/PGDiff/results/real_restoration/s0.0015-seed4321/6bffb069fbc043abf5cbd8323d22ae77_00.png")   

# resize optional
ref = cv2.resize(ref, (src.shape[1], src.shape[0]))


result = color_transfer(src, ref)


cv2.imwrite("result_color_transfer.png", result)
