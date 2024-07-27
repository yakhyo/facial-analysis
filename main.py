import cv2
import argparse
import warnings
import numpy as np

from models import SCRFD, Attribute
from utils.helpers import Face, draw_face_info

warnings.filterwarnings("ignore")


def load_models(detection_model_path: str, attribute_model_path: str):
    """Loads the detection and attribute models.
    Args:
        detection_model_path (str): Path to the detection model file.
        attribute_model_path (str): Path to the attribute model file.
    Returns
        tuple: A tuple containing the detection model and the attribute model.

    """
    try:
        detection_model = SCRFD(model_path=detection_model_path)
        attribute_model = Attribute(model_path=attribute_model_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        raise
    return detection_model, attribute_model


def inference_image(detection_model, attribute_model, image_path, save_output):
    """Processes a single image for face detection and attributes.
    Args:
        detection_model (SCRFD): The face detection model.
        attribute_model (Attribute): The attribute detection model.
        image_path (str): Path to the input image.
        save_output (str): Path to save the output image.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to load image")
        return

    process_frame(detection_model, attribute_model, frame)
    if save_output:
        cv2.imwrite(save_output, frame)
    cv2.imshow("FaceDetection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def inference_video(detection_model, attribute_model, video_source, save_output):
    """Processes a video source for face detection and attributes.
    Args:
        detection_model (SCRFD): The face detection model.
        attribute_model (Attribute): The attribute detection model.
        video_source (str or int): Path to the input video file or camera index.
        save_output (str): Path to save the output video.
    """
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Failed to open video source")
        return

    out = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_output, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(detection_model, attribute_model, frame)
        if save_output:
            out.write(frame)

        cv2.imshow("FaceDetection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()


def process_frame(detection_model, attribute_model, frame):
    """Detects faces and attributes in a frame and draws the information.
    Args:
        detection_model (SCRFD): The face detection model.
        attribute_model (Attribute): The attribute detection model.
        frame (np.ndarray): The image frame to process.
    """
    boxes_list, points_list = detection_model.detect(frame)

    for boxes, keypoints in zip(boxes_list, points_list):
        *bbox, conf_score = boxes
        gender, age = attribute_model.get(frame, bbox)
        face = Face(kps=keypoints, bbox=bbox, age=age, gender=gender)
        draw_face_info(frame, face)


def run_face_analysis(detection_weights, attribute_weights, input_source, save_output=None):
    """Runs face detection on the given input source."""
    detection_model, attribute_model = load_models(detection_weights, attribute_weights)

    if isinstance(input_source, str) and input_source.lower().endswith(('.jpg', '.png', '.jpeg')):
        inference_image(detection_model, attribute_model, input_source, save_output)
    else:
        inference_video(detection_model, attribute_model, input_source, save_output)


def main():
    """Main function to run face detection from command line."""
    parser = argparse.ArgumentParser(description="Run face detection on an image or video")
    parser.add_argument(
        '--detection-weights',
        type=str,
        default="weights/det_10g.onnx",
        help='Path to the detection model weights file'
    )
    parser.add_argument(
        '--attribute-weights',
        type=str,
        default="weights/genderage.onnx",
        help='Path to the attribute model weights file'
    )
    parser.add_argument(
        '--source',
        type=str,
        default="assets/in_image.jpg",
        help='Path to the input image or video file or camera index (0, 1, ...)'
    )
    parser.add_argument('--output', type=str, help='Path to save the output image or video')
    args = parser.parse_args()

    run_face_analysis(args.detection_weights, args.attribute_weights, args.source, args.output)


if __name__ == "__main__":
    main()
