'''
    roboflow knife dataset 을 dino finetuning dataset 형식에 맞게 수정
    - [YOLOv8 PyTorch TXT (예: 1 0.716797 0.395833 0.147461 0.279167)] format annotation 을 
    - dino finetuning csv format, 단일 파일 annotation.csv로 변환 
'''

import os, glob 
import csv
from PIL import Image 

RAW_IMAGES_PATH = "raw_data/images/"
RAW_LABEL_PATH = "raw_data/labels/"
PROCESSED_ANNOTATION_PATH = "processed_annotation/"
PROCESSED_ANNOTATION_NAME = "annotation.csv"

def bbox_convert(bbox_xyCwh_relative, image_size):
    xc_r, yc_r, w_r, h_r = bbox_xyCwh_relative
    image_width, image_height = image_size
    xc, yc, w, h = xc_r * image_width, yc_r * image_height, w_r * image_width, h_r * image_height
    x1, y1 = xc-w/2, yc-h/2
    bbox_xy1wh = [x1, y1, w, h]
    bbox_xy1wh = [int(item) for item in bbox_xy1wh]
    return bbox_xy1wh

def read_label_txt(label_file, image_size):
    with open(label_file, 'r') as f:
        content = f.readline().strip()
        values = content.split()
        
        if len(values)!=5:
            label_dict = {
                "class_id": 5,  # means not target class 
                "bbox_xywh": [0.0, 0.0, 0.0, 0.0]
            }
            return label_dict
        
        bbox_xyCwh_relative = [float(values[1]), float(values[2]), float(values[3]), float(values[4])]
        bbox_xy1wh = bbox_convert(bbox_xyCwh_relative, image_size)
        
        label_dict = {
            "class_id": int(values[0]),
            "bbox_xywh": bbox_xy1wh
        }
        
    return label_dict

def check_matching_image(label_file):
    file_name = os.path.splitext(os.path.basename(label_file))[0]
    image_name = file_name + ".jpg"
    image_file = RAW_IMAGES_PATH + image_name
    image_exists = os.path.exists(image_file) 
    with Image.open(image_file) as img:
        image_width, image_height = img.size
    return image_name, image_exists, (image_width, image_height)

def proc_a_label(label_file):
    image_name, image_exists, image_size = check_matching_image(label_file)
    if not image_exists:  
        pass 
    
    label_dict = read_label_txt(label_file, image_size)
    if label_dict["class_id"] != 0:  
        pass
    
    annDict = {
        "label_name": "knife",
        "bbox_x": label_dict["bbox_xywh"][0],
        "bbox_y": label_dict["bbox_xywh"][1],
        "bbox_width": label_dict["bbox_xywh"][2],
        "bbox_height": label_dict["bbox_xywh"][3],
        "image_name": image_name,
        "image_width": image_size[0],
        "image_height": image_size[1]
    }

    return annDict

def save_annotation(annotations):
    output_annotation_filename = PROCESSED_ANNOTATION_PATH + PROCESSED_ANNOTATION_NAME
    with open(output_annotation_filename, mode='w', newline='') as outputfile:
        writer = csv.DictWriter(outputfile, fieldnames=['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'image_name', 'image_width', 'image_height'])
        writer.writeheader() 
        for row in annotations:
            writer.writerow(row) 
    print(f"annotation saved! : {output_annotation_filename}") 

if __name__ == "__main__":
    annotations = []
    label_files = glob.glob(RAW_LABEL_PATH + "*.txt")
    for idx, label_file in enumerate(label_files):
        annDict = proc_a_label(label_file)
        annotations.append(annDict)
    save_annotation(annotations)
