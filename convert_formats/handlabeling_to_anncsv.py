'''
    gernerate bbox annotation (DINO FN form) from images 
'''
import glob, os
import cv2, csv  

# classname & path setting 

CLASSNAME = "knife"
IMAGES_PATH = "images/"
ANNOTATION_PATH = "annotation/annotation.csv"

# select bbox 

def mouse_click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: 
        clicked_points = param["clicked_points"]
        clicked_checker = param["clicked_checker"]
        if not clicked_checker["clicked_start"]: 
            clicked_points["start_point"] = [x, y] 
            clicked_checker["clicked_start"], clicked_checker["clicked_end"] = True, False
        elif not clicked_checker["clicked_end"]: 
            clicked_points["end_point"] = [x, y] 
            clicked_checker["clicked_start"], clicked_checker["clicked_end"] = False, True 

def draw_bbox(image, point1, point2):
    cv2.rectangle(image, (point1[0], point1[1]), (point2[0], point2[1]), (0, 0, 255), 2) 
    return image 
    
def convert_bbox(point1, point2):
    x, y = (point1[0]+point2[0])/2, (point1[1]+point2[1])/2
    w, h = point2[0]-point1[0], point2[1]-point1[1]
    bbox = [x, y, w, h]
    return bbox

def select_bbox(image_file):
    image = cv2.imread(image_file) 
    WINDOW_NAME = f"Select bbox - {image_file}" 
    cv2.namedWindow(WINDOW_NAME) 
    clicked_points = { 
        "start_point": None,
        "end_point": None
    } 
    clicked_checker = { 
        "clicked_start": False,
        "clicked_end": False,
    } 
    cv2.setMouseCallback(WINDOW_NAME, mouse_click_event, param={"clicked_points":clicked_points, "clicked_checker":clicked_checker}) 
    cv2.imshow(WINDOW_NAME, image)
    while True: 
        key = cv2.waitKey(1) & 0xFF
        isBboxCompeleted = clicked_checker["clicked_end"] and not clicked_checker["clicked_start"]

        if isBboxCompeleted:
            image_with_bbox = draw_bbox(image.copy(), clicked_points["start_point"], clicked_points["end_point"])
            cv2.imshow(WINDOW_NAME, image_with_bbox)
        
        if key == ord('q') or key == 27:
            if isBboxCompeleted:
                bbox = convert_bbox(clicked_points["start_point"], clicked_points["end_point"])
                print(f"Bounding box selected : {bbox}")
                cv2.destroyAllWindows()
                return bbox 
            else:
                print("Bounding box not selected. Exiting...")
                cv2.destroyAllWindows()
                return None 
        
# save annotation 

def save_annotation(writer, image_file, bbox): 
    label_name = CLASSNAME 
    bbox_x, bbox_y, bbox_width, bbox_height = bbox
    image_name = os.path.basename(image_file)
    image = cv2.imread(image_file)
    image_height, image_width, _ = image.shape
    annDict = {
        "label_name": label_name,
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "image_name": image_name,
        "image_width": image_width,
        "image_height": image_height
    }
    writer.writerow(annDict)

# main 

if __name__ == "__main__":
    annotation_file = ANNOTATION_PATH
    fieldnames = ["label_name", "bbox_x", "bbox_y", "bbox_width", "bbox_height", "image_name", "image_width", "image_height"]
    with open(annotation_file, 'w', newline='') as annotation_file:
        writer = csv.DictWriter(annotation_file, fieldnames=fieldnames)
        writer.writeheader()
        
        image_files = glob.glob(os.path.join(IMAGES_PATH + "*.png"))
        for image_file in image_files:
            print(f"Opened image : {image_file}") 
            bbox = select_bbox(image_file) 
            save_annotation(writer, image_file, bbox)

        print("... annotation saved!")

