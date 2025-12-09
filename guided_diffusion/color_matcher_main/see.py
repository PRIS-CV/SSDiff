from color_matcher import ColorMatcher, __version__
from color_matcher.io_handler import load_img_file
from color_matcher.normalizer import Normalizer

# read files
img_ref = load_img_file('/data/wenjieli/code/BFSR/PGDiff/results/real_restoration/s0.0015-seed4321/6bffb069fbc043abf5cbd8323d22ae77_00.png')
img_src = load_img_file('/data/wenjieli/code/BFSR/PGDiff/codeformer.png')

# instantiate object and run process
cm = ColorMatcher()
img_res = cm.transfer(src=img_src, ref=img_ref, method='mkl')

# normalize image intensity to 8-bit unsigned integer
img_res = Normalizer(img_res).uint8_norm()

import os
# create save folder
os.makedirs("see", exist_ok=True)

# save each image separately (optional)
from imageio import imwrite
imwrite('see/source.png', img_src)
imwrite('see/reference.png', img_ref)
imwrite('see/result.png', img_res)