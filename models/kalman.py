import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box.
        bbox is in the format [x1, y1, x2, y2]
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],      # state transition matrix
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],  
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([    # measurement function
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        self.kf.R[2:,2:] *= 10.   # measurement uncertainty
        self.kf.P[4:,4:] *= 1000. # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[4:,4:] *= 0.01  # process uncertainty
        
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_bbox = bbox
        
    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
        self.last_bbox = bbox
        
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        try:
            if((self.kf.x[6]+self.kf.x[2])<=0):
                self.kf.x[6] *= 0.0
                
            self.kf.predict()
            self.age += 1
            if(self.time_since_update>0):
                self.hit_streak = 0
            self.time_since_update += 1
            
            # Get bbox prediction from Kalman filter
            bbox = self._convert_x_to_bbox(self.kf.x)
            
            # Ensure it's a proper numpy array
            if not isinstance(bbox, np.ndarray):
                bbox = np.array(bbox)
                
            # Ensure the shape is correct
            if bbox.size != 4:
                print(f"Warning: Kalman prediction has wrong size: {bbox.size}")
                # Return the last bbox if prediction is invalid
                return self.last_bbox
                
            # Append to history and ensure we're returning a list of 4 floats
            self.history.append(bbox)
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        except Exception as e:
            print(f"Error in Kalman prediction: {str(e)}")
            # Fall back to last known bbox if there's an error
            return self.last_bbox
    
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self._convert_x_to_bbox(self.kf.x)
    
    def _convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the center of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    # scale is area
        r = w / float(h) if h > 0 else 1.0
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x):
        """
        Takes a bounding box in the center form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([
            x[0] - w/2.,
            x[1] - h/2.,
            x[0] + w/2.,
            x[1] + h/2.
        ]).reshape((1,4))[0]