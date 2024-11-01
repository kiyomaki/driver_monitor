import time
import cv2
import mediapipe as mp
import numpy as np
from HeadPoseEstimator import HeadPoseEstimator as HeadPoseEst
from attention import AttentionScorer as AttScorer

def initialize_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise Exception("Cannot open camera")
    return cap

def process_frame(frame, Detector, Head_pose, Scorer, fps):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.expand_dims(gray, axis=2)
    gray = np.concatenate([gray, gray, gray], axis=2)
    lms = Detector.process(gray).multi_face_landmarks

    if not lms:
        return frame, None, None, None

    landmarks = get_landmarks(lms)

    frame_det, roll, pitch, yaw = Head_pose.get_pose(frame, landmarks, frame.shape[1::-1])

    if frame_det is not None:
        frame = frame_det

    if roll is not None:
        cv2.putText(frame, "roll:" + str(roll.round(1)[0]), (450, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1, cv2.LINE_AA)
    if pitch is not None:
        cv2.putText(frame, "pitch:" + str(pitch.round(1)[0]), (450, 70), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1, cv2.LINE_AA)
    if yaw is not None:
        cv2.putText(frame, "yaw:" + str(yaw.round(1)[0]), (450, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1, cv2.LINE_AA)

    t_now = time.perf_counter()
    looking_away, look_away_duration = Scorer.eval_scores(t_now, head_roll=roll, head_pitch=pitch, head_yaw=yaw)

    print(f"Looking Away: {looking_away}, Duration: {look_away_duration:.1f}s, roll: {roll}, pitch: {pitch}, yaw: {yaw}")

    if looking_away:
        cv2.putText(frame, f"LOOKING AWAY! {look_away_duration:.1f}s", (10, 320), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

    return frame, roll, pitch, yaw

def get_landmarks(face_landmarks):
    return np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks[0].landmark], dtype=np.float32)

def main():
    camera_id = 0
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)

    Detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )

    Head_pose = HeadPoseEst(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    Scorer = AttScorer(
        t_now=time.perf_counter(),
        roll_thresh=20,
        pitch_thresh=20,
        yaw_thresh=20,
        pose_time_thresh=1.5,  # よそ見時間閾値設定
        verbose=True,
    )

    cap = initialize_camera(camera_id)
    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame from camera/stream end")
            break

        t_now = time.perf_counter()
        elapsed_time = t_now - prev_time
        prev_time = t_now
        fps = np.round(1 / elapsed_time, 3) if elapsed_time > 0 else 0

        frame, roll, pitch, yaw = process_frame(frame, Detector, Head_pose, Scorer, fps)

        if frame is None:
            continue

        cv2.imshow("Press 'q' to terminate", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

