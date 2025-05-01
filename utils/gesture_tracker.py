# utils/gesture_tracker.py
import cv2
import numpy as np
import random
import time

class GestureTracker:
    """
    Class to track hand gestures using OpenCV color-based detection.
    """
    def __init__(self):
        # Parameters for skin color detection in HSV space
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # For tracking gesture trajectories
        self.trajectory = []
        self.landmarks_history = []
        
    def process_frame(self, frame):
        """
        Process a single frame to detect hand contours.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            processed_frame: Frame with annotations
            landmarks: Detected hand landmarks (simplified)
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a binary mask for skin color
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a copy of the frame for drawing
        processed_frame = frame.copy()
        
        landmarks = []
        
        # Process if contours are found
        if contours:
            # Find the largest contour (assumed to be the hand)
            max_contour = max(contours, key=cv2.contourArea)
            
            # Only process if contour is large enough
            if cv2.contourArea(max_contour) > 3000:
                # Draw the contour
                cv2.drawContours(processed_frame, [max_contour], 0, (0, 255, 0), 2)
                
                # Get convex hull
                hull = cv2.convexHull(max_contour)
                cv2.drawContours(processed_frame, [hull], 0, (0, 0, 255), 2)
                
                # Get defects (for finger detection)
                hull = cv2.convexHull(max_contour, returnPoints=False)
                try:
                    defects = cv2.convexityDefects(max_contour, hull)
                except:
                    defects = None
                
                # Extract feature points
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])
                        
                        # Add to landmarks
                        landmarks.append(list(start) + [0])  # Adding z=0 since we're in 2D
                        landmarks.append(list(end) + [0])
                        landmarks.append(list(far) + [0])
                        
                        # Draw points for visualization
                        cv2.circle(processed_frame, start, 5, (0, 255, 0), -1)
                        cv2.circle(processed_frame, end, 5, (0, 255, 0), -1)
                        cv2.circle(processed_frame, far, 5, (0, 0, 255), -1)
                
                # Calculate centroid
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Store centroid in trajectory
                    self.trajectory.append((cx, cy))
                    if len(self.trajectory) > 30:  # Keep only recent positions
                        self.trajectory.pop(0)
                    
                    # Mark the centroid
                    cv2.circle(processed_frame, (cx, cy), 7, (255, 0, 0), -1)
                    
                    # Draw the trajectory
                    for i in range(1, len(self.trajectory)):
                        cv2.line(processed_frame, self.trajectory[i-1], self.trajectory[i], (0, 255, 0), 2)
        
        # Store landmarks for gesture recognition
        if landmarks:
            # Ensure we have a reasonable number of landmarks
            while len(landmarks) < 21:  # Pad to 21 landmarks for compatibility
                landmarks.append([0, 0, 0])
            
            # Limit to 21 landmarks for compatibility with the original model
            landmarks = landmarks[:21]
            
            self.landmarks_history.append(landmarks)
            if len(self.landmarks_history) > 30:  # Keep a history of 30 frames
                self.landmarks_history.pop(0)
        
        return processed_frame, landmarks
    
    def extract_features(self):
        """
        Extract relevant features from the landmarks history for gesture recognition.
        
        Returns:
            features: Dictionary of features extracted from the hand landmarks
        """
        if not self.landmarks_history:
            return None
        
        # Use the most recent landmarks
        current_landmarks = self.landmarks_history[-1]
        
        # Flatten landmarks to a 1D array
        flat_landmarks = np.array(current_landmarks).flatten()
        
        # Calculate velocities if we have enough history
        velocities = []
        if len(self.landmarks_history) > 1:
            prev_landmarks = self.landmarks_history[-2]
            for i in range(min(len(current_landmarks), len(prev_landmarks))):
                curr = current_landmarks[i]
                prev = prev_landmarks[i]
                vel = [curr[j] - prev[j] for j in range(3)]  # x, y, z velocities
                velocities.extend(vel)
        
        # Calculate relative positions to the center point (first landmark)
        relative_positions = []
        if current_landmarks:
            center = current_landmarks[0]
            for lm in current_landmarks[1:]:
                rel_pos = [lm[j] - center[j] for j in range(3)]  # x, y, z relative positions
                relative_positions.extend(rel_pos)
        
        # Calculate simple angles between points
        angles = []
        if len(current_landmarks) >= 5:  # Need at least 5 points for angles
            for i in range(len(current_landmarks)-2):
                # Create vectors between consecutive points
                v1 = np.array(current_landmarks[i+1][:2]) - np.array(current_landmarks[i][:2])
                v2 = np.array(current_landmarks[i+2][:2]) - np.array(current_landmarks[i+1][:2])
                
                # Calculate angle
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * (180.0 / np.pi)
                    angles.append(angle)
        
        # Combine all features
        features = {
            'landmarks': flat_landmarks.tolist() if len(flat_landmarks) > 0 else [],
            'velocities': velocities,
            'relative_positions': relative_positions,
            'angles': angles,
            'trajectory': self.trajectory.copy()
        }
        
        return features
    
    def get_gesture_data(self, duration=3.0):
        """
        Collect gesture data over a specified duration.
        
        Args:
            duration: Duration in seconds to collect gesture data
            
        Returns:
            gesture_data: List of feature dictionaries collected over the duration
        """
        # In a real implementation, this would capture frames from a webcam
        # and process them over the specified duration, returning a list of features
        
        # Since we can't run actual OpenCV capture in this environment,
        # we'll just simulate it for now
        return self.simulate_gesture_data()
    
    def simulate_gesture_data(self, gesture_type=None):
        """
        Simulate gesture data for demonstration purposes.
        
        Args:
            gesture_type: Optional type of gesture to simulate
            
        Returns:
            gesture_data: List of simulated feature dictionaries
        """
        # Define different simulated trajectory patterns based on gesture type
        trajectories = {
            'Wave': self._simulate_wave_gesture,
            'Pinch': self._simulate_pinch_gesture,
            'Open Hand': self._simulate_open_hand_gesture,
            'Point': self._simulate_point_gesture,
            'Fist': self._simulate_fist_gesture,
            'Thumbs Up': self._simulate_thumbs_up_gesture,
            'Victory': self._simulate_victory_gesture,
            'Swipe Left': self._simulate_swipe_left_gesture,
            'Swipe Right': self._simulate_swipe_right_gesture
        }
        
        # If no specific gesture, pick a random one
        if not gesture_type or gesture_type not in trajectories:
            gesture_type = random.choice(list(trajectories.keys()))
        
        # Generate simulated data using the appropriate function
        return trajectories[gesture_type]()
    
    def _simulate_wave_gesture(self):
        gesture_data = []
        center_x, center_y = 300, 200
        
        # Simulate 30 frames of data
        for i in range(30):
            # Create oscillating x value for a wave
            x = center_x + 50 * np.sin(i * 0.3)
            y = center_y + 10 * np.sin(i * 0.7)  # Slight vertical movement
            
            # Create simulated landmarks (21 points for a full hand)
            landmarks = []
            for j in range(21):
                # Distribute landmarks around the center point with some variation
                lm_x = x + 20 * np.cos(j * 0.3) + random.uniform(-5, 5)
                lm_y = y + 20 * np.sin(j * 0.3) + random.uniform(-5, 5)
                lm_z = random.uniform(-0.1, 0.1)  # Z value
                landmarks.append([lm_x, lm_y, lm_z])
            
            # Simulate a trajectory
            trajectory = [(center_x + 50 * np.sin(t * 0.3), center_y + 10 * np.sin(t * 0.7)) 
                          for t in range(max(0, i-10), i+1)]
            
            # Create feature dictionary
            features = {
                'landmarks': np.array(landmarks).flatten().tolist(),
                'velocities': [random.uniform(-10, 10) for _ in range(63)],  # 21 landmarks * 3 dimensions
                'relative_positions': [random.uniform(-30, 30) for _ in range(60)],  # 20 landmarks * 3 dimensions
                'angles': [random.uniform(0, 180) for _ in range(10)],  # 10 angle combinations
                'trajectory': trajectory
            }
            
            gesture_data.append(features)
        
        return gesture_data
    
    def _simulate_pinch_gesture(self):
        gesture_data = []
        center_x, center_y = 300, 200
        
        # Simulate 30 frames of data
        for i in range(30):
            # Pinch movement simulation - moving thumb and index finger together
            pinch_factor = abs(15 - i) / 15.0  # 1.0 to 0.0 to 1.0 (open to pinched to open)
            
            # Create simulated landmarks (21 points for a full hand)
            landmarks = []
            for j in range(21):
                # First 5 points are the thumb, next 4 are index finger
                if j < 5:
                    # Thumb movement
                    lm_x = center_x - 20 + 15 * pinch_factor + random.uniform(-3, 3)
                    lm_y = center_y - 10 + 15 * j/4 + random.uniform(-3, 3)
                elif j < 9:
                    # Index finger movement
                    lm_x = center_x + 20 - 15 * pinch_factor + random.uniform(-3, 3)
                    lm_y = center_y - 10 + 15 * (j-5)/4 + random.uniform(-3, 3)
                else:
                    # Other fingers stay relatively still
                    finger_index = (j - 9) // 4  # 0, 1, 2 for the three remaining fingers
                    position_in_finger = (j - 9) % 4  # 0-3 for position along the finger
                    lm_x = center_x + (10 + 10 * finger_index) + random.uniform(-3, 3)
                    lm_y = center_y - 10 + 15 * position_in_finger/4 + random.uniform(-3, 3)
                
                lm_z = random.uniform(-0.1, 0.1)  # Z value
                landmarks.append([lm_x, lm_y, lm_z])
            
            # Simulate a trajectory for thumb tip
            trajectory = []
            for t in range(max(0, i-5), i+1):
                factor = abs(15 - t) / 15.0
                x = center_x - 20 + 15 * factor + random.uniform(-2, 2)
                y = center_y - 10 + 15 + random.uniform(-2, 2)
                trajectory.append((x, y))
            
            # Create feature dictionary
            features = {
                'landmarks': np.array(landmarks).flatten().tolist(),
                'velocities': [random.uniform(-5, 5) for _ in range(63)],  # Smaller velocities for a pinch
                'relative_positions': [random.uniform(-30, 30) for _ in range(60)],
                'angles': [random.uniform(0, 180) for _ in range(10)],
                'trajectory': trajectory
            }
            
            gesture_data.append(features)
        
        return gesture_data
    
    def _simulate_open_hand_gesture(self):
        gesture_data = []
        center_x, center_y = 300, 200
        
        # Simulate 30 frames of data
        for i in range(30):
            # Open hand with slight variations for realism
            
            # Create simulated landmarks (21 points for a full hand)
            landmarks = []
            for j in range(21):
                # Index 0 is the wrist, 1-4 are thumb joints, 5-8 are index finger joints, etc.
                finger_index = j // 4  # 0 for wrist/palm, 1 for thumb, 2 for index, etc.
                position_in_finger = j % 4  # 0-3 for position along the finger
                
                # Wrist/palm landmark
                if j == 0:
                    lm_x = center_x + random.uniform(-5, 5)
                    lm_y = center_y + 40 + random.uniform(-5, 5)
                
                # Thumb (angle different from other fingers)
                elif j < 5:
                    angle = -0.6  # Thumb angles differently
                    length = 15 * position_in_finger
                    lm_x = center_x - 30 + length * np.cos(angle) + random.uniform(-3, 3)
                    lm_y = center_y + 20 + length * np.sin(angle) + random.uniform(-3, 3)
                
                # Other fingers
                else:
                    # Calculate the angle for each finger - spread them apart
                    angle = -0.9 + 0.4 * (finger_index - 1)  # -0.9 to 0.7
                    length = 15 * position_in_finger
                    lm_x = center_x - 10 + 20 * (finger_index - 1) + length * np.cos(angle) + random.uniform(-3, 3)
                    lm_y = center_y + 20 - length * np.sin(angle) + random.uniform(-3, 3)
                
                lm_z = random.uniform(-0.1, 0.1)  # Z value
                landmarks.append([lm_x, lm_y, lm_z])
            
            # Simulate a trajectory that stays relatively centered
            trajectory = [(center_x + random.uniform(-5, 5), center_y + random.uniform(-5, 5)) 
                          for _ in range(min(5, i+1))]
            
            # Create feature dictionary
            features = {
                'landmarks': np.array(landmarks).flatten().tolist(),
                'velocities': [random.uniform(-3, 3) for _ in range(63)],  # Small velocities for an open hand
                'relative_positions': [random.uniform(-40, 40) for _ in range(60)],  # Wider spread for open hand
                'angles': [random.uniform(0, 180) for _ in range(10)],
                'trajectory': trajectory
            }
            
            gesture_data.append(features)
        
        return gesture_data
    
    def _simulate_point_gesture(self):
        gesture_data = []
        center_x, center_y = 300, 200
        
        # Direction of pointing (normalized vector)
        direction_x, direction_y = 0.7, -0.7  # Pointing up-right
        
        # Simulate 30 frames of data
        for i in range(30):
            # Create simulated landmarks (21 points for a full hand)
            landmarks = []
            
            # Wrist position
            wrist_x = center_x + random.uniform(-5, 5)
            wrist_y = center_y + 40 + random.uniform(-5, 5)
            landmarks.append([wrist_x, wrist_y, 0])
            
            # Thumb (partially curled)
            for j in range(1, 5):
                lm_x = wrist_x - 15 + j * 3 + random.uniform(-2, 2)
                lm_y = wrist_y - 10 + j * 3 + random.uniform(-2, 2)
                lm_z = random.uniform(-0.1, 0.1)
                landmarks.append([lm_x, lm_y, lm_z])
            
            # Index finger (extended)
            for j in range(1, 5):
                lm_x = wrist_x + j * 10 * direction_x + random.uniform(-2, 2)
                lm_y = wrist_y + j * 10 * direction_y + random.uniform(-2, 2)
                lm_z = random.uniform(-0.1, 0.1)
                landmarks.append([lm_x, lm_y, lm_z])
            
            # Other fingers (curled)
            for finger in range(3):  # Middle, ring, pinky
                for j in range(1, 5):
                    lm_x = wrist_x + 10 + (finger * 5) + (j * 2) * (1-j/5) + random.uniform(-2, 2)
                    lm_y = wrist_y - 5 + (j * 2) * (1-j/5) + random.uniform(-2, 2)
                    lm_z = random.uniform(-0.1, 0.1)
                    landmarks.append([lm_x, lm_y, lm_z])
            
            # Simulate trajectory for the pointing finger tip
            trajectory = []
            for t in range(max(0, i-5), i+1):
                x = wrist_x + 40 * direction_x + random.uniform(-3, 3)
                y = wrist_y + 40 * direction_y + random.uniform(-3, 3)
                trajectory.append((x, y))
            
            # Create feature dictionary
            features = {
                'landmarks': np.array(landmarks).flatten().tolist(),
                'velocities': [random.uniform(-5, 5) for _ in range(63)],
                'relative_positions': [random.uniform(-30, 30) for _ in range(60)],
                'angles': [random.uniform(0, 180) for _ in range(10)],
                'trajectory': trajectory
            }
            
            gesture_data.append(features)
        
        return gesture_data
    
    def _simulate_fist_gesture(self):
        gesture_data = []
        center_x, center_y = 300, 200
        
        # Simulate 30 frames of data
        for i in range(30):
            # Create simulated landmarks (21 points for a full hand)
            landmarks = []
            
            # Wrist position
            wrist_x = center_x + random.uniform(-5, 5)
            wrist_y = center_y + 30 + random.uniform(-5, 5)
            landmarks.append([wrist_x, wrist_y, 0])
            
            # All fingers curled into a fist
            for finger in range(5):  # Thumb, index, middle, ring, pinky
                for j in range(1, 5):
                    # Fingers curl inward
                    curl_factor = j / 4.0  # 0.25 to 1.0
                    
                    # Different positions for different fingers in the fist
                    offset_x = -15 + finger * 8
                    offset_y = -10
                    
                    # Apply curl
                    lm_x = wrist_x + offset_x + (4-j) * 2 + random.uniform(-2, 2)
                    lm_y = wrist_y + offset_y + (4-j) * 2 + random.uniform(-2, 2)
                    lm_z = random.uniform(-0.1, 0.1)
                    landmarks.append([lm_x, lm_y, lm_z])
            
            # Simulate a trajectory that moves slightly
            trajectory = []
            t_range = min(5, i+1)
            for t in range(t_range):
                x = wrist_x + random.uniform(-3, 3)
                y = wrist_y - 10 + random.uniform(-3, 3)
                trajectory.append((x, y))
            
            # Create feature dictionary
            features = {
                'landmarks': np.array(landmarks).flatten().tolist(),
                'velocities': [random.uniform(-3, 3) for _ in range(63)],  # Small velocities for a fist
                'relative_positions': [random.uniform(-20, 20) for _ in range(60)],  # Closer together for a fist
                'angles': [random.uniform(0, 180) for _ in range(10)],
                'trajectory': trajectory
            }
            
            gesture_data.append(features)
        
        return gesture_data
    
    def _simulate_thumbs_up_gesture(self):
        gesture_data = []
        center_x, center_y = 300, 200
        
        # Simulate 30 frames of data
        for i in range(30):
            # Create simulated landmarks (21 points for a full hand)
            landmarks = []
            
            # Wrist position
            wrist_x = center_x + random.uniform(-5, 5)
            wrist_y = center_y + 30 + random.uniform(-5, 5)
            landmarks.append([wrist_x, wrist_y, 0])
            
            # Thumb extended upward
            for j in range(1, 5):
                lm_x = wrist_x - 5 + j * 1 + random.uniform(-2, 2)
                lm_y = wrist_y - j * 10 + random.uniform(-2, 2)  # Moving upward
                lm_z = random.uniform(-0.1, 0.1)
                landmarks.append([lm_x, lm_y, lm_z])
            
            # Other fingers curled
            for finger in range(1, 5):  # Index, middle, ring, pinky
                for j in range(1, 5):
                    # Different positions for different fingers
                    offset_x = -5 + finger * 8
                    offset_y = 0
                    
                    # Apply curl
                    lm_x = wrist_x + offset_x + (4-j) * 2 + random.uniform(-2, 2)
                    lm_y = wrist_y + offset_y + j * 2 + random.uniform(-2, 2)
                    lm_z = random.uniform(-0.1, 0.1)
                    landmarks.append([lm_x, lm_y, lm_z])
            
            # Simulate a trajectory for the thumb tip
            trajectory = []
            for t in range(max(0, i-5), i+1):
                x = wrist_x - 5 + 4 * 1 + random.uniform(-3, 3)
                y = wrist_y - 4 * 10 + random.uniform(-3, 3)
                trajectory.append((x, y))
            
            # Create feature dictionary
            features = {
                'landmarks': np.array(landmarks).flatten().tolist(),
                'velocities': [random.uniform(-5, 5) for _ in range(63)],
                'relative_positions': [random.uniform(-30, 30) for _ in range(60)],
                'angles': [random.uniform(0, 180) for _ in range(10)],
                'trajectory': trajectory
            }
            
            gesture_data.append(features)
        
        return gesture_data
    
    def _simulate_victory_gesture(self):
        gesture_data = []
        center_x, center_y = 300, 200
        
        # Simulate 30 frames of data
        for i in range(30):
            # Create simulated landmarks (21 points for a full hand)
            landmarks = []
            
            # Wrist position
            wrist_x = center_x + random.uniform(-5, 5)
            wrist_y = center_y + 40 + random.uniform(-5, 5)
            landmarks.append([wrist_x, wrist_y, 0])
            
            # Thumb (partially curled)
            for j in range(1, 5):
                lm_x = wrist_x - 15 + j * 3 + random.uniform(-2, 2)
                lm_y = wrist_y - 5 + j * 2 + random.uniform(-2, 2)
                lm_z = random.uniform(-0.1, 0.1)
                landmarks.append([lm_x, lm_y, lm_z])
            
            # Index and middle fingers extended in a V shape
            # Index finger
            for j in range(1, 5):
                angle = -0.3  # Angled outward
                length = j * 10
                lm_x = wrist_x + length * np.cos(angle) + random.uniform(-2, 2)
                lm_y = wrist_y - length * np.sin(angle) + random.uniform(-2, 2)
                lm_z = random.uniform(-0.1, 0.1)
                landmarks.append([lm_x, lm_y, lm_z])
            
            # Middle finger
            for j in range(1, 5):
                angle = 0.3  # Angled outward in other direction
                length = j * 10
                lm_x = wrist_x + length * np.cos(angle) + random.uniform(-2, 2)
                lm_y = wrist_y - length * np.sin(angle) + random.uniform(-2, 2)
                lm_z = random.uniform(-0.1, 0.1)
                landmarks.append([lm_x, lm_y, lm_z])
            
            # Ring and pinky fingers curled
            for finger in range(3, 5):  # Ring, pinky
                for j in range(1, 5):
                    lm_x = wrist_x + 5 + (finger - 3) * 8 + j * 2 * (1-j/4) + random.uniform(-2, 2)
                    lm_y = wrist_y + j * 2 * (1-j/4) + random.uniform(-2, 2)
                    lm_z = random.uniform(-0.1, 0.1)
                    landmarks.append([lm_x, lm_y, lm_z])
            
            # Simulate trajectories for the extended finger tips
            trajectory = []
            for t in range(max(0, i-3), i+1):
                # Index finger tip
                angle1 = -0.3
                length1 = 4 * 10
                x1 = wrist_x + length1 * np.cos(angle1) + random.uniform(-3, 3)
                y1 = wrist_y - length1 * np.sin(angle1) + random.uniform(-3, 3)
                
                # Middle finger tip
                angle2 = 0.3
                length2 = 4 * 10
                x2 = wrist_x + length2 * np.cos(angle2) + random.uniform(-3, 3)
                y2 = wrist_y - length2 * np.sin(angle2) + random.uniform(-3, 3)
                
                # Use average of both finger positions
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                trajectory.append((x, y))
            
            # Create feature dictionary
            features = {
                'landmarks': np.array(landmarks).flatten().tolist(),
                'velocities': [random.uniform(-5, 5) for _ in range(63)],
                'relative_positions': [random.uniform(-40, 40) for _ in range(60)],
                'angles': [random.uniform(0, 180) for _ in range(10)],
                'trajectory': trajectory
            }
            
            gesture_data.append(features)
        
        return gesture_data
    
    def _simulate_swipe_left_gesture(self):
        gesture_data = []
        
        # Start position (right side)
        start_x, start_y = 500, 200
        
        # End position (left side)
        end_x, end_y = 100, 200
        
        # Simulate 30 frames of data
        for i in range(30):
            # Calculate position along the swipe path
            progress = min(1.0, i / 20.0)  # Complete swipe in 20 frames
            current_x = start_x + (end_x - start_x) * progress
            current_y = start_y + (end_y - start_y) * progress
            
            # Create simulated landmarks (21 points for a full hand)
            landmarks = []
            
            # Wrist position
            wrist_x = current_x + random.uniform(-5, 5)
            wrist_y = current_y + 30 + random.uniform(-5, 5)
            landmarks.append([wrist_x, wrist_y, 0])
            
            # All fingers extended but close together (like a flat hand)
            for finger in range(5):  # Thumb, index, middle, ring, pinky
                for j in range(1, 5):
                    # Different positions for different fingers
                    offset_x = -10 + finger * 5  # Fingers close together
                    offset_y = -5 - j * 8  # Extended outward
                    
                    lm_x = wrist_x + offset_x + random.uniform(-2, 2)
                    lm_y = wrist_y + offset_y + random.uniform(-2, 2)
                    lm_z = random.uniform(-0.1, 0.1)
                    landmarks.append([lm_x, lm_y, lm_z])
            
            # Simulate a trajectory for the swipe
            trajectory = []
            for t in range(max(0, i-10), i+1):
                t_progress = min(1.0, t / 20.0)
                x = start_x + (end_x - start_x) * t_progress + random.uniform(-5, 5)
                y = start_y + (end_y - start_y) * t_progress + random.uniform(-5, 5)
                trajectory.append((x, y))
            
            # Create feature dictionary
            features = {
                'landmarks': np.array(landmarks).flatten().tolist(),
                'velocities': [random.uniform(-15, 15) for _ in range(63)],  # Larger velocities for a swipe
                'relative_positions': [random.uniform(-30, 30) for _ in range(60)],
                'angles': [random.uniform(0, 180) for _ in range(10)],
                'trajectory': trajectory
            }
            
            gesture_data.append(features)
        
        return gesture_data
    
    def _simulate_swipe_right_gesture(self):
        gesture_data = []
        
        # Start position (left side)
        start_x, start_y = 100, 200
        
        # End position (right side)
        end_x, end_y = 500, 200
        
        # Simulate 30 frames of data
        for i in range(30):
            # Calculate position along the swipe path
            progress = min(1.0, i / 20.0)  # Complete swipe in 20 frames
            current_x = start_x + (end_x - start_x) * progress
            current_y = start_y + (end_y - start_y) * progress
            
            # Create simulated landmarks (21 points for a full hand)
            landmarks = []
            
            # Wrist position
            wrist_x = current_x + random.uniform(-5, 5)
            wrist_y = current_y + 30 + random.uniform(-5, 5)
            landmarks.append([wrist_x, wrist_y, 0])
            
            # All fingers extended but close together (like a flat hand)
            for finger in range(5):  # Thumb, index, middle, ring, pinky
                for j in range(1, 5):
                    # Different positions for different fingers
                    offset_x = -10 + finger * 5  # Fingers close together
                    offset_y = -5 - j * 8  # Extended outward
                    
                    lm_x = wrist_x + offset_x + random.uniform(-2, 2)
                    lm_y = wrist_y + offset_y + random.uniform(-2, 2)
                    lm_z = random.uniform(-0.1, 0.1)
                    landmarks.append([lm_x, lm_y, lm_z])
            
            # Simulate a trajectory for the swipe
            trajectory = []
            for t in range(max(0, i-10), i+1):
                t_progress = min(1.0, t / 20.0)
                x = start_x + (end_x - start_x) * t_progress + random.uniform(-5, 5)
                y = start_y + (end_y - start_y) * t_progress + random.uniform(-5, 5)
                trajectory.append((x, y))
            
            # Create feature dictionary
            features = {
                'landmarks': np.array(landmarks).flatten().tolist(),
                'velocities': [random.uniform(-15, 15) for _ in range(63)],  # Larger velocities for a swipe
                'relative_positions': [random.uniform(-30, 30) for _ in range(60)],
                'angles': [random.uniform(0, 180) for _ in range(10)],
                'trajectory': trajectory
            }
            
            gesture_data.append(features)
        
        return gesture_data