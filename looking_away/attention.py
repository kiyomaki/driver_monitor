import time

class AttentionScorer:
    def __init__(self, t_now, roll_thresh, pitch_thresh, yaw_thresh, pose_time_thresh, verbose=False):
        self.roll_thresh = roll_thresh
        self.pitch_thresh = pitch_thresh
        self.yaw_thresh = yaw_thresh
        self.pose_time_thresh = pose_time_thresh
        self.last_time_attended = t_now
        self.distracted_time = 0.0
        self.verbose = verbose

    def eval_scores(self, t_now, head_roll, head_pitch, head_yaw):
        looking_away = False

        if (
            (head_roll is not None and abs(head_roll) > self.roll_thresh)
            or (head_pitch is not None and abs(head_pitch) > self.pitch_thresh)
            or (head_yaw is not None and abs(head_yaw) > self.yaw_thresh)
        ):
            self.distracted_time = t_now - self.last_time_attended
        else:
            self.last_time_attended = t_now
            self.distracted_time = 0.0

        if self.distracted_time >= self.pose_time_thresh:
            looking_away = True

        if self.verbose:
            print(f"looking away: {looking_away}, distracted_time: {self.distracted_time:.1f}s")

        return looking_away, self.distracted_time

