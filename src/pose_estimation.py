import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import argparse
import os

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
    POSE_CONNECTIONS = [(0,1), (0,2), (1,3), (2,4), (0,5), (0,6), (5,7), (7,9), (6,8), (8,10),
                       (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]
    
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

def process_video(video_path, model, output_path):
    """Processa um vídeo frame por frame para detecção de pose"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return False
    
    # Obter propriedades do vídeo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Configurar vídeo de saída
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {total_frames} frames at {fps} FPS")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Executar inferência no frame
        results = model(frame, size=960)
        
        # Extrair keypoints
        keypoints_list = extract_keypoints(results)
        
        if keypoints_list:
            print(f"Frame {frame_count}: Detected {len(keypoints_list)} persons with pose keypoints")
            # Desenhar pose no frame
            draw_pose(frame, keypoints_list)
        else:
            print(f"Frame {frame_count}: No pose keypoints detected")
        
        # Escrever frame processado
        out.write(frame)
        frame_count += 1
        
        # Progresso a cada 10 frames
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    out.release()
    print(f"Video processing completed. Output saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="YOLOv7 Pose Estimation")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--weights", type=str, default="yolov7-w6-pose.pt", help="Path to YOLOv7 pose weights")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to output image")
    args = parser.parse_args()

    # Validate input arguments
    if not args.image and not args.video:
        print("Error: Either --image or --video argument must be provided")
        return
    
    if args.image and args.video:
        print("Error: Only one of --image or --video can be provided")
        return
    
    # Check if input file exists
    input_path = args.image if args.image else args.video
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found")
        return
    
    # Determine if input is image or video
    is_video = args.video is not None

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model using torch.hub (same as demo.py)
    if not os.path.exists(args.weights):
        print(f"Weights file '{args.weights}' not found, downloading...")
        os.system(
            f"wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{os.path.basename(args.weights)} -O {args.weights}"
        )
    
    try:
        model = torch.hub.load("WongKinYiu/yolov7:pose", "custom", args.weights, trust_repo=True)
        model.to(device)
        model.conf = 0.25  # confidence threshold
        model.iou = 0.65   # IoU threshold
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Processar imagem ou vídeo
    if not is_video:
        # Processar imagem
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image '{args.image}'")
            return
        
        # Executar inferência
        print("Running inference on image...")
        results = model(image, size=960)
        
        # Extrair keypoints
        keypoints_list = extract_keypoints(results)
        
        if keypoints_list:
            print(f"Detected {len(keypoints_list)} persons with pose keypoints")
            # Desenhar pose na imagem
            draw_pose(image, keypoints_list)
        else:
            print("No pose keypoints detected in the image")
        
        nimg = image

        # Salvar saída
        cv2.imwrite(args.output, nimg)
        print(f"Output saved to: {args.output}")

        # Exibir resultado
        plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.imshow(cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB))
        plt.show()
    else:
        # Processar vídeo
        process_video(args.video, model, args.output)

if __name__ == "__main__":
    main()