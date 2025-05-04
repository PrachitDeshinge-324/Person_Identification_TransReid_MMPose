# models/bytetrack.py
import numpy as np
from collections import deque

class ByteTrack:
    """
    Minimal ByteTrack implementation for multi-object tracking.
    This is a simplified version for integration with YOLOv8 detections.
    """
    def __init__(self, track_thresh=0.5, match_thresh=0.8, max_age=30, min_hits=3):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 1

    class Track:
        def __init__(self, bbox, conf, track_id):
            self.bbox = bbox
            self.conf = conf
            self.track_id = track_id
            self.age = 0
            self.hits = 1
            self.time_since_update = 0
            self.history = deque(maxlen=30)

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def update(self, detections):
        # detections: list of (bbox, conf)
        updated_tracks = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        matches = []
        # Match detections to tracks by IoU
        if self.tracks:
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t, track in enumerate(self.tracks):
                for d, (bbox, conf) in enumerate(detections):
                    iou_matrix[t, d] = self.iou(track.bbox, bbox)
            while True:
                if iou_matrix.size == 0:
                    break
                t, d = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if iou_matrix[t, d] < self.match_thresh:
                    break
                matches.append((t, d))
                iou_matrix[t, :] = -1
                iou_matrix[:, d] = -1
                unmatched_tracks.remove(t)
                unmatched_dets.remove(d)
        # Update matched tracks
        for t, d in matches:
            bbox, conf = detections[d]
            track = self.tracks[t]
            track.bbox = bbox
            track.conf = conf
            track.hits += 1
            track.time_since_update = 0
            track.history.append(bbox)
            updated_tracks.append(track)
        # Create new tracks for unmatched detections
        for d in unmatched_dets:
            bbox, conf = detections[d]
            if conf < self.track_thresh:
                continue
            track = self.Track(bbox, conf, self.next_id)
            self.next_id += 1
            updated_tracks.append(track)
        # Age unmatched tracks
        for t in unmatched_tracks:
            track = self.tracks[t]
            track.time_since_update += 1
            if track.time_since_update < self.max_age:
                updated_tracks.append(track)
        # Remove dead tracks
        self.tracks = [t for t in updated_tracks if t.time_since_update < self.max_age]
        # Return active tracks with sufficient hits
        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits or track.time_since_update == 0:
                results.append((track.track_id, track.bbox, track.conf))
        return results
