"""
    ai hub dataset 을 dino finetuning dataset 형식에 맞게 수정
    - ai hub 원본 데이터는 video(mp4) / label(json) 형식
    - video에서 event 범위에 해당하는 smokingman 정보를 사용
    - 이를 dino training form 에 맞게 image(png) / annotation(csv, 단일) 형식으로 변환
"""

import cv2 
import json, csv 
import os, glob 
import re 

# data path / class name settings 

RAW_VIDEO_PATH = "raw_data/video/"
RAW_LABEL_PATH = "raw_data/label/"
PROCESSED_IMAGE_PATH = "processed_data/images/"
PROCESSED_ANNOTATION_PATH = "processed_data/annotation/"
PROCESSED_ANNOTATION_NAME = "annotation.csv"
CLASS_NAME = "smokingPerson"

# raw json parsers 

def info_parser(info):
    video_name = info["filename"]  
    frame_size = [info["width"], info["height"]]
    return video_name, frame_size

def events_parser(events):
    event_frames = []
    for event in events: 
        if event["object_id"] == 1:
            ev_start_frame = event["ev_start_frame"]
            ev_end_frame = event["ev_end_frame"]
            event_frames.append([ev_start_frame, ev_end_frame])
    return event_frames

def annotation_parser(annotation): 
    cur_frame = annotation["cur_frame"] 
    bbox_raw = annotation["bbox"]
    x1, y1 = bbox_raw[0]
    x2, y2 = bbox_raw[1]
    bbox_x, bbox_y = (x1+x2)/2, (y1+y2)/2
    bbox_width, bbox_height = x2-x1, y2-y1 
    bbox = [bbox_x, bbox_y, bbox_width, bbox_height] 
    return cur_frame, bbox

def annotations_parser(annotations):
    bboxes = []
    annotations = [annotation for annotation in annotations if annotation["class_name"]=="smoking"]
    annotations = sorted(annotations, key=lambda x: x["cur_frame"])  
    for annotation in annotations:
        cur_frame, bbox = annotation_parser(annotation)
        bboxes.append([cur_frame, bbox])
    return bboxes

# processing  

def generate_image_name_base(video_name): 
    """ 
        video_name 예 : C_1_31_jap_cl_09-01_17-16-00_b_set_DF2.mp4 
        -> 17-16-00_b 와 같은 부분을 추출하여, 대응되는 image name의 식별코드로 사용 
        image_name 예 : 171600b_168.png 
    """  
    match = re.search(r'(\d{2}-\d{2}-\d{2}_[a-z])', video_name)
    if match: 
        extracted_part = match.group(1).replace('-', '')
        image_name_base = extracted_part
        return image_name_base
    else:
        print(f"{video_name}에서 '17-16-00_b'와 같은 식별 코드를 찾을 수 없습니다. ")
        return None 

def process_annotation(video_name, frame_size, bboxes):
    processed_annotations = [] 
    
    image_name_base = generate_image_name_base(video_name) 
    image_width, image_height = frame_size
    label_name = CLASS_NAME 
    
    for cur_frame, bbox in bboxes: 
        image_name = f"{image_name_base}_{cur_frame}.png" 
        bbox_x_center, bbox_y_center, bbox_width, bbox_height = bbox 
        bbox_x, bbox_y = bbox_x_center-bbox_width/2, bbox_y_center-bbox_height/2

        frame_annotation = {
            "label_name": label_name,
            "bbox_x": int(bbox_x),
            "bbox_y": int(bbox_y),
            "bbox_width": int(bbox_width),
            "bbox_height": int(bbox_height),
            "image_name": image_name,
            "image_width": image_width,
            "image_height": image_height
        }
        processed_annotations.append(frame_annotation)
    
    return processed_annotations 

# saving 

def save_annotation(processed_annotations):
    output_annotation_filename = PROCESSED_ANNOTATION_PATH + PROCESSED_ANNOTATION_NAME
    with open(output_annotation_filename, mode='w', newline='') as outputfile:
        writer = csv.DictWriter(outputfile, fieldnames=['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'image_name', 'image_width', 'image_height'])
        writer.writeheader() 
        for row in processed_annotations:
            writer.writerow(row) 
    print(f"annotation saved! : {output_annotation_filename}") 

def save_images(video_name, event_frames): 
    cap = cv2.VideoCapture(RAW_VIDEO_PATH + video_name) 
    
    if not cap.isOpened():
        print(f"Cannot open '{video_name}' file.")
        exit() 

    while True: 
        ret, frame = cap.read()
        if not ret: 
            break
        cur_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if any(start <= cur_frame_num <= end for start, end in event_frames):
            OUTPUTIMG_BASENAME = generate_image_name_base(video_name)
            output_image_name = f"{OUTPUTIMG_BASENAME}_{cur_frame_num}.png"
            cv2.imwrite(PROCESSED_IMAGE_PATH + output_image_name, frame)

    print(f"images all saved! : from {video_name}, frame range {event_frames}") 
    cap.release()    

# check file matching 

def check_rawfiles_matching():
    label_files = glob.glob(RAW_LABEL_PATH+'*.json')
    label_filenames = [os.path.splitext(os.path.basename(f))[0] for f in label_files] 
    video_files = glob.glob(RAW_VIDEO_PATH+'*.mp4')
    video_filenames = [os.path.splitext(os.path.basename(f))[0] for f in video_files] 
    if set(label_filenames) == set(video_filenames) :
        print("check : raw_labels and raw_videos are matching correctly!", end="\n.\n.\n")
    else:
        print("check : raw_labels and raw_videos are not matching...", end="\n.\n.\n")

def check_savefiles_matching():
    saved_image_files = glob.glob(PROCESSED_IMAGE_PATH+'*.png')
    saved_image_filenames = [os.path.basename(f) for f in saved_image_files] 
    saved_annotation_filename = PROCESSED_ANNOTATION_PATH + PROCESSED_ANNOTATION_NAME
    with open(saved_annotation_filename) as annfile:
        ann_reader = csv.DictReader(annfile)
        ann_image_filenames = [row["image_name"] for row in ann_reader]
    if set(saved_image_filenames) == set(ann_image_filenames) :
        print("check : images and annotation image names are matching correctly!", end="\n.\n.\n")
    else:
        print("check : images and annotation image names are not matching...", end="\n.\n.\n")

# main

if __name__=="__main__": 
    # check raw_data file matching : label - video 
    check_rawfiles_matching()
    
    # load raw label files 
    processed_annotations = []
    raw_label_files = glob.glob(RAW_LABEL_PATH + '*.json')
    for raw_label_file in raw_label_files:
        print(f"## now processing {os.path.basename(raw_label_file)} file ##")
        with open(raw_label_file, "r", encoding="utf-8") as jsonfile: 
            raw_label_json = json.load(jsonfile) 

        # parse json raw label file 
        video_name, frame_size = info_parser(raw_label_json["info"])
        event_frames = events_parser(raw_label_json["events"])
        bboxes = annotations_parser(raw_label_json["annotations"]) 

        # process annotation & save event images  
        processed_annotations.extend(process_annotation(video_name, frame_size, bboxes)) 
        save_images(video_name, event_frames)
    
    # save annotation 
    save_annotation(processed_annotations)    
    # check processed data file matching : annotation - images 
    check_savefiles_matching()
