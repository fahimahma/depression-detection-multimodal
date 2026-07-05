import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class FacialFeatureExtractor:
    """
    Extracts facial features for depression detection including:
    - Facial action units (AUs)
    - Facial expressions
    - Eye gaze patterns
    - Head pose estimation
    """
    
    def __init__(self):
        try:
            import dlib
            self.dlib_detector = dlib.get_frontal_face_detector()
            # Load shape predictor (68 facial landmarks)
            self.shape_predictor = dlib.shape_predictor(
                'models/shape_predictor_68_face_landmarks.dat'
            )
        except Exception as e:
            print(f"Warning: Could not load dlib models: {e}")
            self.dlib_detector = None
            self.shape_predictor = None
        
        # Initialize OpenCV face detector
        self.cascade_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Eye cascade
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def extract_from_video(self, video_path: str, 
                          sample_every_n_frames: int = 5) -> Dict[str, float]:
        """
        Extract facial features from video file.
        
        Args:
            video_path: Path to video file
            sample_every_n_frames: Sample every n frames to reduce computation
            
        Returns:
            Dictionary containing aggregated facial features
        """
        cap = cv2.VideoCapture(video_path)
        
        frame_features_list = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_every_n_frames == 0:
                frame_features = self.extract_from_frame(frame)
                if frame_features:
                    frame_features_list.append(frame_features)
            
            frame_count += 1
        
        cap.release()
        
        # Aggregate features across frames
        return self._aggregate_frame_features(frame_features_list)
    
    def extract_from_frame(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Extract facial features from single frame.
        
        Args:
            frame: Image frame (BGR format)
            
        Returns:
            Dictionary containing facial features or None if no face detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.cascade_classifier.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        features = {}
        
        # Process first face
        x, y, w, h = faces[0]
        face_region = frame[y:y+h, x:x+w]
        gray_face = gray[y:y+h, x:x+w]
        
        # Facial landmarks
        features.update(self._extract_landmarks(gray_face, faces[0], gray))
        
        # Eye features
        features.update(self._extract_eye_features(face_region, gray_face))
        
        # Head pose
        features.update(self._extract_head_pose(gray_face, faces[0], gray))
        
        # Skin features
        features.update(self._extract_skin_features(face_region))
        
        # Expression features
        features.update(self._extract_expression_features(face_region))
        
        return features
    
    def _extract_landmarks(self, gray_face: np.ndarray, face_rect: Tuple, 
                          gray: np.ndarray) -> Dict[str, float]:
        """Extract facial landmarks and related features."""
        features = {}
        
        if self.shape_predictor is None:
            return {f'landmark_{i}': 0.0 for i in range(5)}
        
        try:
            import dlib
            x, y, w, h = face_rect
            dlib_rect = dlib.rectangle(x, y, x + w, y + h)
            
            landmarks = self.shape_predictor(gray, dlib_rect)
            
            # Convert landmarks to numpy array
            landmark_points = np.array([(p.x, p.y) for p in landmarks.parts()])
            
            # Mouth features
            mouth_points = landmark_points[48:68]  # Mouth region
            mouth_height = np.mean([mouth_points[i, 1] for i in [2, 9]])
            mouth_width = mouth_points[-1, 0] - mouth_points[0, 0]
            features['landmark_mouth_openness'] = float(mouth_height / (mouth_width + 1e-10))
            
            # Eye features
            left_eye = landmark_points[36:42]   # Left eye
            right_eye = landmark_points[42:48]  # Right eye
            
            left_eye_aspect = self._eye_aspect_ratio(left_eye)
            right_eye_aspect = self._eye_aspect_ratio(right_eye)
            
            features['landmark_left_eye_aspect'] = float(left_eye_aspect)
            features['landmark_right_eye_aspect'] = float(right_eye_aspect)
            features['landmark_eye_aspect_avg'] = float((left_eye_aspect + right_eye_aspect) / 2)
            
            # Eyebrow features
            left_brow = landmark_points[17:22]  # Left eyebrow
            right_brow = landmark_points[22:27] # Right eyebrow
            
            features['landmark_eyebrow_distance'] = float(
                np.mean([left_eye[:, 1]]) - np.mean([left_brow[:, 1]])
            )
            
            # Nose features
            nose_tip = landmark_points[30]  # Nose tip
            features['landmark_nose_x'] = float(nose_tip[0])
            features['landmark_nose_y'] = float(nose_tip[1])
            
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
        
        return features
    
    def _extract_eye_features(self, face_region: np.ndarray, 
                             gray_face: np.ndarray) -> Dict[str, float]:
        """Extract eye-related features."""
        features = {}
        
        try:
            eyes = self.eye_cascade.detectMultiScale(gray_face)
            features['eye_count'] = float(len(eyes))
            
            if len(eyes) > 0:
                # Eye area size
                eye_areas = [w * h for (x, y, w, h) in eyes]
                features['eye_area_mean'] = float(np.mean(eye_areas))
                features['eye_area_std'] = float(np.std(eye_areas)) if len(eye_areas) > 1 else 0.0
            else:
                features['eye_area_mean'] = 0.0
                features['eye_area_std'] = 0.0
            
            # Gaze direction (simplified)
            features['eye_openness'] = self._estimate_eye_openness(face_region)
            
        except Exception as e:
            print(f"Error extracting eye features: {e}")
        
        return features
    
    def _extract_head_pose(self, gray_face: np.ndarray, face_rect: Tuple,
                          gray: np.ndarray) -> Dict[str, float]:
        """Extract head pose features."""
        features = {}
        
        try:
            x, y, w, h = face_rect
            
            # Head pose estimation based on face bounds
            face_center_x = x + w / 2
            face_center_y = y + h / 2
            frame_center_x = gray.shape[1] / 2
            frame_center_y = gray.shape[0] / 2
            
            # Yaw (left-right)
            yaw = (face_center_x - frame_center_x) / (gray.shape[1] / 2)
            features['head_pose_yaw'] = float(yaw)
            
            # Pitch (up-down)
            pitch = (face_center_y - frame_center_y) / (gray.shape[0] / 2)
            features['head_pose_pitch'] = float(pitch)
            
            # Face size (proxy for distance)
            face_area = w * h
            frame_area = gray.shape[0] * gray.shape[1]
            features['head_pose_distance'] = float(face_area / frame_area)
            
        except Exception as e:
            print(f"Error extracting head pose: {e}")
        
        return features
    
    def _extract_skin_features(self, face_region: np.ndarray) -> Dict[str, float]:
        """Extract skin-related features."""
        features = {}
        
        try:
            # Convert to HSV for skin analysis
            hsv_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            
            # Skin color statistics
            h_channel = hsv_face[:, :, 0]
            s_channel = hsv_face[:, :, 1]
            v_channel = hsv_face[:, :, 2]
            
            features['skin_hue_mean'] = float(np.mean(h_channel))
            features['skin_hue_std'] = float(np.std(h_channel))
            features['skin_saturation_mean'] = float(np.mean(s_channel))
            features['skin_value_mean'] = float(np.mean(v_channel))
            
            # Skin roughness (from edge detection)
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_face, 100, 200)
            features['skin_roughness'] = float(np.sum(edges) / (face_region.shape[0] * face_region.shape[1]))
            
        except Exception as e:
            print(f"Error extracting skin features: {e}")
        
        return features
    
    def _extract_expression_features(self, face_region: np.ndarray) -> Dict[str, float]:
        """Extract facial expression features."""
        features = {}
        
        try:
            # Convert to LAB color space
            lab_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            
            # Brightness (L channel)
            l_channel = lab_face[:, :, 0]
            features['expression_brightness_mean'] = float(np.mean(l_channel))
            features['expression_brightness_std'] = float(np.std(l_channel))
            
            # Texture analysis
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Local Binary Pattern (LBP) - simplified
            laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
            features['expression_texture_variance'] = float(np.var(laplacian))
            
            # Motion (temporal difference would need multiple frames)
            features['expression_contrast'] = float(np.std(gray_face) / (np.mean(gray_face) + 1e-10))
            
        except Exception as e:
            print(f"Error extracting expression features: {e}")
        
        return features
    
    @staticmethod
    def _eye_aspect_ratio(eye_points: np.ndarray) -> float:
        """Calculate eye aspect ratio (EAR) to detect if eyes are open/closed."""
        # Distance between vertical eye landmarks
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Distance between horizontal eye landmarks
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C) if C > 0 else 0.0
        return ear
    
    @staticmethod
    def _estimate_eye_openness(face_region: np.ndarray) -> float:
        """Estimate eye openness from face region."""
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Get upper half of face (where eyes are)
        eye_region = gray_face[:gray_face.shape[0]//2, :]
        
        # Threshold to find dark regions (pupils)
        _, binary = cv2.threshold(eye_region, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Calculate ratio of dark pixels
        dark_pixels = np.sum(binary > 0)
        total_pixels = binary.shape[0] * binary.shape[1]
        
        return float(dark_pixels / (total_pixels + 1e-10))
    
    @staticmethod
    def _aggregate_frame_features(frame_features_list: List[Dict]) -> Dict[str, float]:
        """Aggregate features across multiple frames."""
        if not frame_features_list:
            return {}
        
        aggregated = {}
        
        # Get all feature keys
        all_keys = set()
        for frame_dict in frame_features_list:
            all_keys.update(frame_dict.keys())
        
        # Compute statistics across frames
        for key in all_keys:
            values = [f[key] for f in frame_features_list if key in f]
            
            if values:
                aggregated[f'{key}_mean'] = float(np.mean(values))
                aggregated[f'{key}_std'] = float(np.std(values))
                aggregated[f'{key}_min'] = float(np.min(values))
                aggregated[f'{key}_max'] = float(np.max(values))
        
        return aggregated
