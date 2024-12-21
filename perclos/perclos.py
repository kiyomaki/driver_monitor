import time
import pprint
import cv2
import mediapipe as mp
import numpy as np
from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from parser import get_args
from utils import get_landmarks, load_camera_parameters
from csv_handler import CSVHandler

def initialize_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise Exception("Cannot open camera")
    return cap

def process_frame(frame, Detector, Eye_det, Scorer, args, camera_matrix, dist_coeffs, fps):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.expand_dims(gray, axis=2)
    gray = np.concatenate([gray, gray, gray], axis=2)
    lms = Detector.process(gray).multi_face_landmarks

    if not lms:
        return frame, None, None, None

    landmarks = get_landmarks(lms)
    Eye_det.show_eye_keypoints(color_frame=frame, landmarks=landmarks, frame_size=frame.shape[1::-1])
    ear = Eye_det.get_EAR(frame=gray, landmarks=landmarks)

    if ear is not None:
        cv2.putText(frame, "EAR:" + str(round(ear, 3)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

    t_now = time.perf_counter()
    _, perclos_score = Scorer.get_PERCLOS(t_now, fps, ear)
    asleep = Scorer.eval_scores(t_now, ear_score=ear)

    print(f"PERCLOS Score: {round(perclos_score, 3)}")
    print(f"Asleep: {asleep}")

    if asleep:
        cv2.putText(frame, "ASLEEP!", (10, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

    # PERCLOSスコア表示
    cv2.putText(frame, "PERCLOS:" + str(round(perclos_score, 3)), (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

    return frame, ear, perclos_score, asleep

def main():
    args = get_args()

    # ユーザーにCSVファイル名を入力させる
    output_file = CSVHandler.get_user_filename()
    csv_handler = CSVHandler(output_file)
    csv_handler.write_header()

    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)
        except:
            print("OpenCV optimization could not be set to True, the script may be slower than expected")

    camera_matrix, dist_coeffs = load_camera_parameters(args.camera_params) if args.camera_params else (None, None)

    if args.verbose:
        print("Arguments and Parameters used:\n")
        pprint.pp(vars(args), indent=4)
        print("\nCamera Matrix:")
        pprint.pp(camera_matrix, indent=4)
        print("\nDistortion Coefficients:")
        pprint.pp(dist_coeffs, indent=4)
        print("\n")

    Detector = mp.solutions.face_mesh.FaceMesh( # type: ignore
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )

    Eye_det = EyeDet(show_processing=args.show_eye_proc)
    Scorer = AttScorer(
        t_now=time.perf_counter(),
        ear_thresh=args.ear_thresh,
        ear_time_thresh=args.ear_time_thresh,
        perclos_thresh=0.02,  # 閾値設定
        # gaze_thresh=args.gaze_thresh,
        # roll_thresh=args.roll_thresh,
        # pitch_thresh=args.pitch_thresh,
        # yaw_thresh=args.yaw_thresh,
        gaze_time_thresh=args.gaze_time_thresh,
        pose_time_thresh=args.pose_time_thresh,
        verbose=args.verbose,
    )

    cap = initialize_camera(args.camera)
    prev_time = time.perf_counter()
    prev_state = None
    last_output_time = time.time()
    start_time = time.time()

    perclos_accumulator = []  # PERCLOS Scoreリスト生成

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame from camera/stream end")
            break

        t_now = time.perf_counter()
        elapsed_time = t_now - prev_time
        prev_time = t_now
        fps = np.round(1 / elapsed_time, 3) if elapsed_time > 0 else 0

        frame, ear, perclos_score, asleep = process_frame(frame, Detector, Eye_det, Scorer, args, camera_matrix, dist_coeffs, fps)
        if frame is None:
            continue

        if perclos_score is not None:
            perclos_accumulator.append(perclos_score)  # PERCLOS Scoreをリストに追加

        current_state = "Asleep" if asleep else "Awake"
        elapsed_seconds = int(time.time() - start_time)  # 開始時刻からの経過秒数

        # 1秒ごとに結果を出力
        if time.time() - last_output_time >= 1:
            if len(perclos_accumulator) > 0:
                average_perclos = sum(perclos_accumulator) / len(perclos_accumulator)
                csv_handler.append_row(elapsed_seconds, round(average_perclos, 3), current_state)
                print(f"[Output] Elapsed Seconds: {elapsed_seconds}, Average PERCLOS: {round(average_perclos, 3)}, State: {current_state}")
            else:
                csv_handler.append_row(elapsed_seconds, "", "")  # 空欄の行を出力
                print(f"[Output] Elapsed Seconds: {elapsed_seconds}, No Data")
            perclos_accumulator.clear()  
            last_output_time = time.time()

        if args.show_fps:
            cv2.putText(frame, "FPS:" + str(round(fps)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)
        if args.show_proc_time:
            proc_time_frame_ms = ((cv2.getTickCount() - prev_time) / cv2.getTickFrequency()) * 1000
            cv2.putText(frame, "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + "ms", (10, 430), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)

        cv2.imshow("Press 'q' to terminate", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
