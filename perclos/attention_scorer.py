class AttentionScorer:
    def __init__(self, t_now, ear_thresh, gaze_thresh, perclos_thresh=0.03, 
                 # roll_thresh=20, pitch_thresh=20, yaw_thresh=20,
                 ear_time_thresh=0.3, gaze_time_thresh=2.0, pose_time_thresh=4.0, verbose=False):
        self.PERCLOS_TIME_PERIOD = 60
        self.ear_thresh = ear_thresh = 0.2
        self.perclos_thresh = perclos_thresh
        self.ear_time_thresh = ear_time_thresh

        self.last_time_eye_opened = t_now
        self.prev_time = t_now

        self.closure_time = 0
        self.eye_closure_counter = 0

        self.verbose = verbose

    def eval_scores(self, t_now, ear_score):
        asleep = False

        if self.closure_time >= self.ear_time_thresh:
            asleep = True

        if (ear_score is not None) and (ear_score <= self.ear_thresh):
            self.closure_time = t_now - self.last_time_eye_opened
        elif ear_score is None or (ear_score is not None and ear_score > self.ear_thresh):
            self.last_time_eye_opened = t_now
            self.closure_time = 0.0

        if self.verbose:
            print(f"eye closed:{asleep}")

        return asleep

    def get_PERCLOS(self, t_now, fps, ear_score):
        delta = t_now - self.prev_time
        tired = False

        all_frames_numbers_in_perclos_duration = int(self.PERCLOS_TIME_PERIOD * fps)

        if (ear_score is not None) and (ear_score <= self.ear_thresh):
            self.eye_closure_counter += 1

        perclos_score = self.eye_closure_counter / all_frames_numbers_in_perclos_duration

        if perclos_score >= self.perclos_thresh:
            tired = True

        if (ear_score is not None) and (ear_score > self.ear_thresh):
            self.eye_closure_counter = 0
            self.prev_time = t_now

        return tired, perclos_score
