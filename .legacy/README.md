# Legacy Codes 

## Convert Formats 

`./convert_formats/*` 는 Net Challenge Project에서 사용된 데이터셋 전처리 코드들이다.

이 프로젝트는 K-Digital Net Challenge (2023)에서 수행되었으며,
YOLOv8, AIHub JSON 등 다양한 어노테이션 형식을 GroundingDINO 파인튜닝에 사용할 수 있는 통합(annotation) 포맷으로 변환하기 위한 데이터 전처리 스크립트들을 포함하고 있다. 

<!-- A project from the K-Digital Net Challenge (2023). 
This repository contains data preprocessing scripts, designed to convert various labeling formats (e.g., YOLOv8, AIHub JSON) into a unified annotation format for fine-tuning GroundingDINO.  -->

아래는 실행 방법에 대한 설명이다: 

To run the format converting, you have to move on `./convert_formats/` directory, and follow the below running guide. 

- hand labeling to annotation.csv :  
    1. prepare folders : 
        - images/
        - annotation/ 
    2. run 
        ```bash 
        cd ./convert_formats/
        python handlabeling_to_anncsv.py
        ```

- yolov8 labeling(txt) to annotation.csv : 
    1. prepare folders : 
        - raw_data/images/ 
        - raw_data/labels/
        - processed_annotation/
    2. run 
        ```bash 
        cd ./convert_formats/
        python yolotxt_to_anncsv.py 
        ```

- custom dataset (video, json) to annotation.csv
    (here we used aihub smoking person dataset)
    1. prepare folders : 
        - raw_data/video/
        - raw_data/label/
        - processed_data/images/
        - processed_data/annotation/
    2. run 
        ```bash 
        cd ./convert_formats/
        python aihub_to_anncsv.py 
        ```

- custom dataset (video, json) to yolov8 lageling(txt) : 
    1. prepare folders : 
        - raw_data/video/
        - raw_data/label/
        - processed_data/images/
        - processed_data/label/
    2. run 
        ```bash 
        cd ./convert_formats/
        python aihub_to_yolo/aihub_to_yolov8txt.py
        ```



--- 