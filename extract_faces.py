import os
import cv2
import argparse
import glob
import torch
from basicsr.utils import imwrite
from basicsr.utils.misc import get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default='/data/wenjieli/code/BFSR/face_with_mask',
                        help='Input image path or folder.')
    parser.add_argument('-o', '--output_path', type=str,default='/data/wenjieli/code/BFSR/face_with_mask/face',
                        help='Output folder for aligned faces.')
    parser.add_argument('--only_center_face', action='store_true',
                        help='Only extract center face.')
    parser.add_argument('--detection_model', type=str, default='retinaface_resnet50',
                        help='Face detector: retinaface_resnet50 | retinaface_mobile0.25 | YOLOv5l | dlib')
    args = parser.parse_args()

    device = get_device()
    face_helper = FaceRestoreHelper(
        1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=args.detection_model,
        save_ext='png',
        use_parse=True,
        device=device
    )

    os.makedirs(args.output_path, exist_ok=True)

    if os.path.isfile(args.input_path):
        input_img_list = [args.input_path]
    else:
        input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
        if len(input_img_list) == 0:
            raise FileNotFoundError(f'No images found in {args.input_path}')

    for i, img_path in enumerate(input_img_list):
        print(f'[{i+1}/{len(input_img_list)}] Processing: {img_path}')
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)

        face_helper.clean_all()
        face_helper.read_image(img)
        num_faces = face_helper.get_face_landmarks_5(only_center_face=args.only_center_face, resize=640)

        if num_faces == 0:
            print(f'  No face detected in {img_name}')
            continue

        face_helper.align_warp_face()

        for idx, aligned_face in enumerate(face_helper.cropped_faces):
            face_save_name = f'{img_name}_{idx:02d}.png'
            save_path = os.path.join(args.output_path, face_save_name)
            imwrite(aligned_face, save_path)
            print(f'  Saved: {save_path}')

    print(f'\n All extracted faces saved in: {args.output_path}')


if __name__ == '__main__':
    main()
