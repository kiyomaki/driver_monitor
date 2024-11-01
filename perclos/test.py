import time
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from parser import get_args
from utils import get_landmarks, load_camera_parameters

def initialize_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise Exception("Cannot open camera")
    return cap

def process_frame(frame, Detector, Eye_det, Scorer, args, camera_matrix, dist_coeffs, fps, perclos_threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.expand_dims(gray, axis=2)
    gray = np.concatenate([gray, gray, gray], axis=2)
    lms = Detector.process(gray).multi_face_landmarks

    if not lms:
        return frame, None, None, time.perf_counter()  # Ensure to return four values

    landmarks = get_landmarks(lms)
    Eye_det.show_eye_keypoints(color_frame=frame, landmarks=landmarks, frame_size=frame.shape[1::-1])
    ear = Eye_det.get_EAR(frame=gray, landmarks=landmarks)

    if ear is not None:
        cv2.putText(frame, f"EAR: {round(ear, 3)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

    t_now = time.perf_counter()
    _, perclos_score = Scorer.get_PERCLOS(t_now, fps, ear)
    asleep = Scorer.eval_scores(t_now, ear_score=ear)

    # Debug information output
    print(f"PERCLOS Score: {perclos_score:.3f}")
    print(f"Asleep: {asleep}")

    # Display the PERCLOS score and sleep warning on the frame
    cv2.putText(frame, f"PERCLOS: {perclos_score:.3f}", (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
    if asleep:
        cv2.putText(frame, "ASLEEP!", (10, 130), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

    return frame, ear, perclos_score, t_now

def main():
    args = get_args()
    camera_matrix, dist_coeffs = load_camera_parameters(args.camera_params) if args.camera_params else (None, None)
    Detector = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)
    Eye_det = EyeDet(show_processing=args.show_eye_proc)
    Scorer = AttScorer(time.perf_counter(), args.ear_thresh, args.gaze_thresh, 0.02, args.roll_thresh, args.pitch_thresh, args.yaw_thresh, args.ear_time_thresh, args.gaze_time_thresh, args.pose_time_thresh, args.verbose)
    cap = initialize_camera(args.camera)
    prev_time = time.perf_counter()
    perclos_values = []
    time_stamps = []
    perclos_threshold = 0.2  # Set your desired threshold here
    threshold_exceeded = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame from camera/stream end")
            break

        t_now = time.perf_counter()
        elapsed_time = t_now - prev_time
        prev_time = t_now
        fps = np.round(1 / elapsed_time, 3) if elapsed_time > 0 else 0

        frame, ear, perclos_score, current_time = process_frame(frame, Detector, Eye_det, Scorer, args, camera_matrix, dist_coeffs, fps, perclos_threshold)
        if frame is None:
            continue

        perclos_values.append(perclos_score)
        time_stamps.append(current_time)

        # Check if PERCLOS score exceeds the threshold
        if perclos_score > perclos_threshold:
            threshold_exceeded.append(len(time_stamps) - 1)  # Store the index of the exceeded frame

        # Display the frame
        cv2.imshow("Press 'q' to terminate", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Plot PERCLOS over time
    plt.figure()
    plt.plot(time_stamps, perclos_values, label='PERCLOS')

    # Highlight points where PERCLOS exceeds the threshold
    for idx in threshold_exceeded:
        plt.plot(time_stamps[idx], perclos_values[idx], 'ro')  # 'ro' for red circle marker

    plt.xlabel('Time (s)')
    plt.ylabel('PERCLOS')
    plt.title('PERCLOS Over Time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

