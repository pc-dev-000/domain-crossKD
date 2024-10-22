import argparse
import os
import glob
import time
import cv2
import numpy as np
import darknet


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    parser.add_argument("--nms_thresh", type=float, default=0.5,
                        help="remove duplicating detections")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    assert 0 < args.nms_thresh < 1, "NMS Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))

def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def image_detection(image_or_path, network, class_names, class_colors, thresh):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    if type(image_or_path) == "str":
        input_image = cv2.imread(image_or_path)
    else:
        input_image = image_or_path
    input_image = cv2.imread(image_or_path)
    image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), input_image, detections

def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height

COLORS = np.random.uniform(0, 255, size=(4, 3))
COLORS[0] = [255,0,0]
COLORS[1] = [0,255,0]
COLORS[2] = [0,0,255]
COLORS[3] = [0,255,255]

size_limiter = np.array([[700], [500], [450], [1000]])

def calculate_iou(box1, box2):
    """
    Calculate IoU score between two bounding boxes
    """
    boxA = []
    x1, y1, w1, h1 = box1
    boxA.append(x1)
    boxA.append(y1)
    boxA.append(x1 + w1)
    boxA.append(y1 + h1)

    boxB = []
    x2, y2, w2, h2 = box2
    boxB.append(x2)
    boxB.append(y2)
    boxB.append(x2 + w2)
    boxB.append(y2 + h2)

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max((xB - xA), 0) * max((yB - yA), 0)
    if interArea == 0:
        return 0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_itr(box1, box2):
    """
    Calculate intersection score between two bounding boxes
    """
    boxA = []
    x1, y1, w1, h1 = box1
    boxA.append(x1)
    boxA.append(y1)
    boxA.append(x1 + w1)
    boxA.append(y1 + h1)

    boxB = []
    x2, y2, w2, h2 = box2
    boxB.append(x2)
    boxB.append(y2)
    boxB.append(x2 + w2)
    boxB.append(y2 + h2)

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max((xB - xA), 0) * max((yB - yA), 0) / (w1 * h1)
    return interArea

def rp_fusion(box1, box2):
    """
    Fuse bounding boxes between rider and pedestrian
    """
    boxA = []
    x1, y1, w1, h1 = box1
    boxA.append(x1)
    boxA.append(y1)
    boxA.append(x1 + w1)
    boxA.append(y1 + h1)

    boxB = []
    x2, y2, w2, h2 = box2
    boxB.append(x2)
    boxB.append(y2)
    boxB.append(x2 + w2)
    boxB.append(y2 + h2)

    xA = min(boxA[0], boxB[0])
    yA = min(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])
    return xA, yA, (xB - xA), (yB - yA)

def show4itri(image, input_image, class_index, class_name, input_bbox):
    """
    Limit FOV based on predefined Autosys layout
    """     
    color = COLORS[class_index]
    x1 = input_bbox[0]
    y1 = input_bbox[1]
    x2 = input_bbox[0] + input_bbox[2]
    y2 = input_bbox[1] + input_bbox[3]
    cv2.rectangle(input_image, (x1, y1), (x2, y2), color, 1)
    cv2.putText(input_image, class_name, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def convert2itri(image, input_image, bbox):
    """
    Convert YOLOv7x coordinates into Autosys coordinates
    """
    x, y, w, h = bbox
    model_height, model_width, _ = image.shape
    image_height, image_width, _ = input_image.shape
    ratio_height = image_height/model_height
    ratio_width = image_width/model_width
    return round(x*ratio_width), round(y*ratio_height), round(w*ratio_width), round(h*ratio_height)

def post4itri(image, input_image, detections, class_names, thresh, nms_thresh):
    c_index = []
    c_names = []
    c_confs = []
    c_boxes = []
    for label, confidence, bbox in detections:
        label = class_names.index(label)
        x, y, w, h = convert2itri(image, input_image, bbox)
        match label:
            case 0:
                class_name = "person"
            case 1:
                class_name = "rider"
            case 2:
                class_name = "2-wheels"
            case 3:
                class_name = "4-wheels"
            case _:
                class_name = "unknown"
        c_index.append(label)
        c_names.append(class_name)
        c_confs.append(float(confidence))
        c_boxes.append([round(x - (w / 2)), round(y - (h / 2)), w, h])

    image_height, image_width, _ = input_image.shape
    filter_ratio = (image_height * image_width) / 579840
    nms_indices = cv2.dnn.NMSBoxes(c_boxes, c_confs, thresh, nms_thresh)
    total_detection = 0
    fov_indices = []
    for obj in nms_indices:
        box_dim = c_boxes[obj][2] * c_boxes[obj][3]
        filter_dim = np.round(size_limiter[c_index[obj]] / filter_ratio).astype(int)
        if (box_dim >= filter_dim):
            total_detection += 1
            fov_indices.append(obj)
            show4itri(image, input_image, c_index[obj], c_names[obj], c_boxes[obj])
    return c_index, c_boxes, fov_indices, total_detection

def save_annotations(name, input_image, c_index, c_boxes, fov_indices, total_detection):
    """
    Files saved with image_name.txt and relative coordinates
    """
    #file_name = os.path.splitext(name)[0] + "_debug.jpg"
    #cv2.imwrite(file_name, input_image)
    image_height, image_width, _ = input_image.shape
    file_path = os.path.splitext(name)[0]
    out_txt_path = file_path.replace("GroundTruths/RGB", "Predictions/TXT")
    out_vis_path = file_path.replace("GroundTruths/RGB", "Predictions/VIS")
    out_vis_name = out_vis_path + ".jpg"
    cv2.imwrite(out_vis_name, input_image)
    out_txt_name = out_txt_path + ".txt"
    print(out_txt_name)
    with open(out_txt_name, "w") as f:
        f.write("{:d}\n".format(total_detection))
        for obj in fov_indices:
            x1 = c_boxes[obj][0]
            y1 = c_boxes[obj][1]
            x2 = c_boxes[obj][0] + c_boxes[obj][2]
            y2 = c_boxes[obj][1] + c_boxes[obj][3]
            c = c_index[obj]
            f.write("{:d} {:d} {:d} {:d} {:d}\n".format(c, x1, y1, x2, y2))

def main():
    args = parser()
    check_arguments_errors(args)

    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights
    )

    images = load_images(args.input)
    index = 0
    while True:
        if args.input:
            if index >= len(images):
                break
            image_name = images[index]
        else:
            image_name = input("Enter Image Path: ")
        print(image_name)
        prev_time = time.time()
        image, input_image, detections = image_detection(
            image_name, network, class_names, class_colors, args.thresh)
        classes, boxes, indices, n_detection = post4itri(
            image, input_image, detections, class_names, args.thresh, args.nms_thresh)
        if args.save_labels:
            save_annotations(image_name, input_image, classes, boxes, indices, n_detection)

        fps = int(1 / (time.time() - prev_time))
        print("FPS: {}".format(fps))
        if not args.dont_show:
            cv2.imshow('Inference', input_image)
            #cv2.imwrite("inference.png", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        index += 1


if __name__ == "__main__":
    main()
