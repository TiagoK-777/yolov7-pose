import argparse
import os
from typing import List, Optional, Union

import numpy as np
import cv2
import torch
import torchvision.ops.boxes as bops

import norfair
from norfair import Detection, Paths, Tracker, Video

DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000

POSE_CONNECTIONS = [(0,1), (0,2), (1,3), (2,4), (0,5), (0,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]


class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if not os.path.exists(model_path):
            os.system(
                f"wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{os.path.basename(model_path)} -O {model_path}"
            )

        # load model
        try:
            self.model = torch.hub.load("WongKinYiu/yolov7:pose", "custom", model_path, trust_repo=True)
        except:
            raise Exception("Failed to load model from {}".format(model_path))

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


def center(points):
    return [np.mean(np.array(points), axis=0)]


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(
                    points=centroid,
                    scores=scores,
                    label=int(detection_as_xywh[-1].item()),
                )
            )
    elif track_points == "bbox":
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )
            scores = np.array(
                [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
            )
            norfair_detections.append(
                Detection(
                    points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item())
                )
            )

    return norfair_detections


def extract_keypoints(yolo_detections):
    """Extract pose keypoints from YOLOv7 pose detections"""
    keypoints_list = []

    # Check if keypoints are available
    if hasattr(yolo_detections, 'keypoints') and yolo_detections.keypoints is not None:
        keypoints_data = yolo_detections.keypoints
        
        # Process each detection's keypoints
        for i in range(len(keypoints_data)):
            person_keypoints = keypoints_data[i]
            kp_list = []
            
            for kp in person_keypoints:
                x, y, conf = kp
                if conf > 0.3:  # confidence threshold for keypoints
                    kp_list.append((int(x), int(y)))
                else:
                    kp_list.append(None)
            
            keypoints_list.append(kp_list)
    
    return keypoints_list


def draw_pose(frame, keypoints_list):
    """Draw pose keypoints and skeleton connections on frame"""
    for keypoints in keypoints_list:
        # Draw keypoints
        for kp in keypoints:
            if kp is not None:
                cv2.circle(frame, kp, 5, (0, 255, 0), -1)  # Green keypoints
        
        # Draw skeleton connections
        for conn in POSE_CONNECTIONS:
            if (len(keypoints) > max(conn) and
                keypoints[conn[0]] is not None and
                keypoints[conn[1]] is not None):
                cv2.line(frame, keypoints[conn[0]], keypoints[conn[1]], (255, 0, 0), 2)  # Blue connections


parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument("files", type=str, nargs="+", help="Video files to process")
parser.add_argument(
    "--detector-path", type=str, default="/yolov7-w6-pose.pt", help="YOLOv7 pose model path"
)
parser.add_argument(
    "--img-size", type=int, default="720", help="YOLOv7 inference size (pixels)"
)
parser.add_argument(
    "--conf-threshold",
    type=float,
    default="0.25",
    help="YOLOv7 object confidence threshold",
)
parser.add_argument(
    "--iou-threshold", type=float, default="0.45", help="YOLOv7 IOU threshold for NMS"
)
parser.add_argument(
    "--classes",
    nargs="+",
    type=int,
    help="Filter by class: --classes 0, or --classes 0 2 3",
)
parser.add_argument(
    "--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'"
)
parser.add_argument(
    "--track-points",
    type=str,
    default="bbox",
    help="Track points: 'centroid' or 'bbox'",
)
parser.add_argument(
    "--pose",
    action="store_true",
    help="Enable pose estimation and drawing",
)
args = parser.parse_args()

model = YOLO(args.detector_path, device=args.device)

for input_path in args.files:
    video = Video(input_path=input_path)

    distance_function = "iou" if args.track_points == "bbox" else "euclidean"

    distance_threshold = (
        DISTANCE_THRESHOLD_BBOX
        if args.track_points == "bbox"
        else DISTANCE_THRESHOLD_CENTROID
    )

    tracker = Tracker(
        distance_function=distance_function,
        distance_threshold=distance_threshold,
    )

    for frame in video:
        yolo_detections = model(
            frame,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            image_size=args.img_size,
            classes=args.classes,
        )
        keypoints = extract_keypoints(yolo_detections)
        detections = yolo_detections_to_norfair_detections(
            yolo_detections, track_points=args.track_points
        )
        tracked_objects = tracker.update(detections=detections)
        if args.track_points == "centroid":
            norfair.draw_points(frame, detections)
            norfair.draw_tracked_objects(frame, tracked_objects)
        elif args.track_points == "bbox":
            norfair.draw_boxes(frame, detections)
            norfair.draw_tracked_boxes(frame, tracked_objects)
        
        # Draw pose keypoints if enabled and detected
        if args.pose:
            if keypoints:
                print(f"Detected {len(keypoints)} persons with pose keypoints")
                draw_pose(frame, keypoints)
            else:
                print("No pose keypoints detected in this frame")
        
        video.write(frame)
