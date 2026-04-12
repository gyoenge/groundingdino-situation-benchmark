"""
    ai hub dataset 을 dino finetuning dataset 형식에 맞게 수정
    - ai hub 원본 데이터는 video(mp4) / label(json) 형식
    - video에서 event 범위에 해당하는 smokingman 정보를 사용
    - 이를 [YOLOv8 PyTorch TXT (예: 1 0.716797 0.395833 0.147461 0.279167)] format annotation 으로 변환 
"""

import os, glob 
import cv2 
import json, re 

# data path settings 
RAW_VIDEO_PATH = "raw_data/video/"
RAW_LABEL_PATH = "raw_data/label/"
RESULT_IMAGES_PATH = "processed_data/images/"
RESULT_LABEL_PATH = "processed_data/label/"

# parser for 'AIHUB smokingperson dataset json label format'  
class JsonLabelParser():  
    def __init__(self, jsonlabel_file): 
        video_exists, video_name = self._check_matching_video(jsonlabel_file)
        if not video_exists:
            raise RuntimeError("MATCHING VIDEO NOT FOUND .. ")
        
        with open(jsonlabel_file, "r", encoding="utf-8") as jsonfile: 
            raw_label_json = json.load(jsonfile)  

        # parse json raw label file 
        self.video_name, self.frame_size = self._info_parser(raw_label_json["info"])
        if video_name != self.video_name :
            raise RuntimeError("VIDEO NAME IN LABEL DOES NOT MATCHING .. ")
        event_frames = self._events_parser(raw_label_json["events"])
        self.event_frames = sorted(event_frames, key=lambda x: x[0])
        self.bboxes = self._annotations_parser(raw_label_json["annotations"]) 
        ### bboxes item example : [259, [0.48, 0.45, 0.06, 0.11]] 
        # : [frame_idx, [xcr,ycr,wr,hr]] 
        ### (+ raw bbox form in jsonlabel) 
        # : [259, [930.8410163879398, 481.18691635131836, 119.28203277587818, 114.37383270263672]] 
        # : [frame_idx, [x1,y1,x2,y2]] 

    def get_label_infos(self):
        label_infos = {
            "video_name": self.video_name,
            "frame_size": self.frame_size,
            "event_frames": self.event_frames,
            "bboxes": self.bboxes            
        }
        return label_infos 
    
    def print_label_infos(self, showDetail=False):
        print("=============================================================================")
        print(f": LABEL INFOS for {self.video_name}    ")
        print(f"# video frame size : {self.frame_size}")
        print(f"# event frames section count : {len(self.event_frames)}")
        merge_frame_sections_str = ""
        for event_frame in self.event_frames: 
            merge_frame_sections_str += f"{event_frame[0]}~{event_frame[1]} "
        print(f"### event frames : {merge_frame_sections_str}")
        print(f"# bboxes total count : {len(self.bboxes)}")
        if showDetail: print(f"### bboxes : {self.bboxes}")
        print("=============================================================================")

    def _check_matching_video(self, label_file): 
        file_name = os.path.splitext(os.path.basename(label_file))[0] 
        video_name = file_name + ".mp4" 
        video_file = RAW_VIDEO_PATH + video_name 
        video_exists = os.path.exists(video_file) 
        return video_exists, video_name

    def _info_parser(self, info): 
        video_name = info["filename"]  
        frame_size = [info["width"], info["height"]]
        return video_name, frame_size

    def _events_parser(self, events):
        event_frames = [] 
        for event in events: 
            if event["object_id"] == 1:
                ev_start_frame = event["ev_start_frame"]
                ev_end_frame = event["ev_end_frame"]
                event_frames.append([ev_start_frame, ev_end_frame])
        return event_frames

    def _annotation_parser(self, annotation): 
        cur_frame = annotation["cur_frame"] 
        bbox_raw = annotation["bbox"]
        x1, y1 = bbox_raw[0]
        x2, y2 = bbox_raw[1]
        xc, yc = (x1+x2)/2, (y1+y2)/2
        w, h = x2-x1, y2-y1 
        frame_w, frame_h = self.frame_size
        bbox = [xc/frame_w, yc/frame_h, w/frame_w, h/frame_h]  ### returns yolo8.bbox format 
        rounded_bbox = [round(num, 2) for num in bbox]
        return cur_frame, rounded_bbox

    def _annotations_parser(self, annotations):
        bboxes = []
        annotations = [annotation for annotation in annotations if annotation["class_name"]=="smoking"]
        annotations = sorted(annotations, key=lambda x: x["cur_frame"])  
        for annotation in annotations:
            cur_frame, bbox = self._annotation_parser(annotation)
            bboxes.append([cur_frame, bbox])
        return bboxes

# dataset maker to 'yolov8 format', from video & {frame:bbox}list 
class DatasetMaker():
    def __init__(self, label_infos, class_id=0, extract_ratio=1.0):  
        # label_infos={"video_name":..,"frame_size":..,"event_frames":..,"bboxes":..}
        
        self.video_name = label_infos["video_name"]
        video_exists, video_file = self._check_matching_video(self.video_name)
        if not video_exists: 
            raise RuntimeError("MATCHING VIDEO NOT FOUND .. ") 

        self.video_file = video_file
        self.event_frames = label_infos["event_frames"]
        self.bbox_list = label_infos["bboxes"]
        self.class_id = class_id
        self.extract_size = int(len(self.bbox_list)*extract_ratio)
        self.extract_step = len(self.bbox_list)//self.extract_size
        self.result_namebase = self.video_name.split(".")[0]
        # self.result_namebase = self._generate_unique_namebase()
    
    def generate_dataset(self): 
        print(f"Generate dataset from {self.video_name} ,, ") 
        cap = cv2.VideoCapture(self.video_file) 
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open '{self.video_file}' file .. ") 

        saving_count = 0
        while True:  
            ret, frame = cap.read()
            if not ret: 
                break 
            cur_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) # 1부터 시작 
            # save 
            if any(start <= cur_frame_num <= end for start, end in self.event_frames):
                if saving_count % self.extract_step == 0: 
                    # print(cur_frame_num, end=",") 
                    ### result image 
                    result_image_file = RESULT_IMAGES_PATH + self.result_namebase + f"_{cur_frame_num}.jpg" 
                    cv2.imwrite(result_image_file, frame)
                    ### result txtlabel 
                    result_txtlabel_file = RESULT_LABEL_PATH + self.result_namebase + f"_{cur_frame_num}.txt"
                    # print(self.bbox_list[cur_frame_num][1])
                    print(saving_count, cur_frame_num)
                    txtlabel = [self.class_id]+(self.bbox_list[saving_count][1])
                    # print(txtlabel) 
                    with open(result_txtlabel_file, "w") as file:
                        file.write(' '.join(map(str, txtlabel)))
                saving_count += 1

        print(f"Dataset saved from {self.video_name}, for frame range {self.event_frames}")
        print(f", with extract step {self.extract_step}, total {self.extract_size} sets saved.") 
        cap.release()    
    
    def _check_matching_video(self, video_name): 
        video_file = RAW_VIDEO_PATH + video_name 
        video_exists = os.path.exists(video_file) 
        return video_exists, video_file
    
    def _generate_unique_namebase(self): 
        """ 
            video_name 예 : C_1_31_jap_cl_09-01_17-16-00_b_set_DF2.mp4 
            -> 17-16-00_b 와 같은 부분을 추출하여, 대응되는 image name의 식별코드로 사용 
            image_name 예 : 171600b_168.png 
        """  
        match = re.search(r'(\d{2}-\d{2}-\d{2}_[a-z])', self.video_name)
        if match: 
            extracted_part = match.group(1).replace('-', '')
            namebase = extracted_part
            return namebase
        else:
            raise RuntimeError(f"{self.video_name}에서 '17-16-00_b'와 같은 식별 코드를 찾을 수 없습니다 .. ")
    
# main 
if __name__ == "__main__":
    # result label & images 폴더 초기화 
    # print("=============================================================================")
    # def delete_files_in_directory(directory_path):
    #     for filename in os.listdir(directory_path):
    #         file_path = os.path.join(directory_path, filename)
    #         if os.path.isfile(file_path):
    #             os.remove(file_path)
    # print(f"** '{RESULT_LABEL_PATH.rstrip('/')}', '{RESULT_IMAGES_PATH.rstrip('/')}' 폴더 초기화")
    # delete_files_in_directory(RESULT_LABEL_PATH.rstrip('/'))
    # delete_files_in_directory(RESULT_IMAGES_PATH.rstrip('/'))
    # 데이터셋 생성 
    print("=============================================================================")
    label_files = glob.glob(RAW_LABEL_PATH + "*.json")
    for idx, label_file in enumerate(label_files):
        print("=============================================================================")
        print(f"rawdata count : {idx+1}")
        parsed_label = JsonLabelParser(label_file)
        #parsed_label.print_label_infos(showDetail=True)
        parsed_label.print_label_infos()
        DatasetMaker(parsed_label.get_label_infos(), class_id=0, extract_ratio=0.01).generate_dataset() 
         
