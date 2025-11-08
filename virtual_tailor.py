#pip install opencv-python ultralytics numpy
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Tuple, Optional
import math

class VirtualTailorMeasurements:
    """
    Real-time Computer Vision system for virtual tailoring measurements using T-Pose detection.
    Uses Ultralytics YOLO for pose estimation with live webcam feed.
    """
    
    def __init__(self, model_name: str = 'yolov8n-pose.pt'):
        """
        Initialize the measurement system.
        
        Args:
            model_name: YOLO pose model to use (default: yolov8n-pose.pt)
        """
        self.model = YOLO(model_name)
        
        # YOLO pose keypoint indices (COCO format)
        self.KEYPOINT_MAP = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }
        
        # State variables
        self.measurements_locked = False
        self.final_measurements = None
        
    def detect_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pose keypoints from video frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Array of keypoints (x, y, confidence) or None if detection fails
        """
        results = self.model(frame, verbose=False)
        
        if len(results) == 0 or results[0].keypoints is None:
            return None
        
        # Get first person detected
        keypoints = results[0].keypoints.data[0].cpu().numpy()
        return keypoints
    
    def calculate_distance(self, pt1: np.ndarray, pt2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    
    def validate_tpose(self, keypoints: np.ndarray, 
                       min_confidence: float = 0.5) -> Tuple[bool, str]:
        """
        Validate if the pose is a proper T-Pose.
        
        Args:
            keypoints: Detected keypoints array
            min_confidence: Minimum confidence threshold for keypoints
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if critical keypoints are detected with sufficient confidence
        critical_points = [
            'left_wrist', 'right_wrist', 'left_shoulder', 'right_shoulder',
            'left_hip', 'right_hip', 'left_ankle', 'right_ankle', 'nose'
        ]
        
        for point_name in critical_points:
            idx = self.KEYPOINT_MAP[point_name]
            if keypoints[idx][2] < min_confidence:
                return False, f"Missing: {point_name.replace('_', ' ').title()}"
        
        # Check if arms are extended (T-Pose validation)
        left_shoulder = keypoints[self.KEYPOINT_MAP['left_shoulder']][:2]
        right_shoulder = keypoints[self.KEYPOINT_MAP['right_shoulder']][:2]
        left_wrist = keypoints[self.KEYPOINT_MAP['left_wrist']][:2]
        right_wrist = keypoints[self.KEYPOINT_MAP['right_wrist']][:2]
        
        # Check if wrists are roughly at shoulder level (horizontal arms)
        left_shoulder_to_wrist_y = abs(left_wrist[1] - left_shoulder[1])
        right_shoulder_to_wrist_y = abs(right_wrist[1] - right_shoulder[1])
        shoulder_width = self.calculate_distance(left_shoulder, right_shoulder)
        
        # Arms should be roughly horizontal (Y difference shouldn't be too large)
        if left_shoulder_to_wrist_y > shoulder_width * 0.6 or right_shoulder_to_wrist_y > shoulder_width * 0.6:
            return False, "Arms not horizontal - extend arms out"
        
        # Check if arms are wide enough (proper T-pose)
        left_arm_width = abs(left_wrist[0] - left_shoulder[0])
        right_arm_width = abs(right_wrist[0] - right_shoulder[0])
        
        if left_arm_width < shoulder_width * 0.5 or right_arm_width < shoulder_width * 0.5:
            return False, "Arms not extended enough - spread wider"
        
        return True, ""
    
    def calculate_measurements(self, keypoints: np.ndarray, 
                               user_height: float, 
                               height_unit: str = 'cm') -> Dict:
        """
        Calculate tailoring measurements from keypoints.
        
        Args:
            keypoints: Detected keypoints array
            user_height: User's actual height
            height_unit: Unit of height measurement
            
        Returns:
            Dictionary containing measurements and confidence scores
        """
        # Calculate scale factor using height
        head_point = keypoints[self.KEYPOINT_MAP['nose']][:2]
        left_ankle = keypoints[self.KEYPOINT_MAP['left_ankle']][:2]
        right_ankle = keypoints[self.KEYPOINT_MAP['right_ankle']][:2]
        ankle_center = (left_ankle + right_ankle) / 2
        
        pixel_height = self.calculate_distance(head_point, ankle_center)
        scale_factor = user_height / pixel_height  # units per pixel
        
        # 1. Wingspan: fingertip to fingertip (using wrists as proxy)
        left_wrist = keypoints[self.KEYPOINT_MAP['left_wrist']][:2]
        right_wrist = keypoints[self.KEYPOINT_MAP['right_wrist']][:2]
        wingspan_pixels = self.calculate_distance(left_wrist, right_wrist)
        wingspan = wingspan_pixels * scale_factor
        wingspan_confidence = min(
            keypoints[self.KEYPOINT_MAP['left_wrist']][2],
            keypoints[self.KEYPOINT_MAP['right_wrist']][2]
        )
        
        # 2. Inseam/Leg Length: crotch to floor
        left_hip = keypoints[self.KEYPOINT_MAP['left_hip']][:2]
        right_hip = keypoints[self.KEYPOINT_MAP['right_hip']][:2]
        crotch = (left_hip + right_hip) / 2
        
        inseam_pixels = self.calculate_distance(crotch, ankle_center)
        inseam = inseam_pixels * scale_factor
        inseam_confidence = min(
            keypoints[self.KEYPOINT_MAP['left_hip']][2],
            keypoints[self.KEYPOINT_MAP['right_hip']][2],
            keypoints[self.KEYPOINT_MAP['left_ankle']][2],
            keypoints[self.KEYPOINT_MAP['right_ankle']][2]
        )
        
        # 3. Stomach Width: side-to-side at waist
        left_shoulder = keypoints[self.KEYPOINT_MAP['left_shoulder']][:2]
        right_shoulder = keypoints[self.KEYPOINT_MAP['right_shoulder']][:2]
        
        waist_left = (left_hip + left_shoulder) / 2
        waist_right = (right_hip + right_shoulder) / 2
        
        stomach_width_pixels = self.calculate_distance(waist_left, waist_right)
        stomach_width = stomach_width_pixels * scale_factor
        stomach_confidence = min(
            keypoints[self.KEYPOINT_MAP['left_hip']][2],
            keypoints[self.KEYPOINT_MAP['right_hip']][2],
            keypoints[self.KEYPOINT_MAP['left_shoulder']][2],
            keypoints[self.KEYPOINT_MAP['right_shoulder']][2]
        )
        
        return {
            'Wingspan': {'value': round(wingspan, 1), 'confidence': round(wingspan_confidence * 100, 1)},
            'Inseam': {'value': round(inseam, 1), 'confidence': round(inseam_confidence * 100, 1)},
            'Stomach Width': {'value': round(stomach_width, 1), 'confidence': round(stomach_confidence * 100, 1)},
            'unit': height_unit
        }
    
    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Draw skeleton overlay on frame."""
        # Define skeleton connections
        skeleton = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Draw connections
        for start_idx, end_idx in skeleton:
            if keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5:
                start_point = tuple(keypoints[start_idx][:2].astype(int))
                end_point = tuple(keypoints[end_idx][:2].astype(int))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw keypoints
        for i, kp in enumerate(keypoints):
            if kp[2] > 0.5:
                cv2.circle(frame, tuple(kp[:2].astype(int)), 5, (0, 0, 255), -1)
        
        return frame
    
    def draw_overlay(self, frame: np.ndarray, is_valid: bool, error_msg: str, 
                    measurements: Optional[Dict] = None) -> np.ndarray:
        """Draw UI overlay on frame."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (w - 10, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Title
        cv2.putText(frame, "VIRTUAL TAILORING SYSTEM", (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        # Status
        if self.measurements_locked:
            status_text = "MEASUREMENTS LOCKED"
            status_color = (0, 255, 0)
        elif is_valid:
            status_text = "T-POSE DETECTED - Hold position..."
            status_color = (0, 255, 0)
        else:
            status_text = f"ALIGN BODY: {error_msg}"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Display measurements
        if measurements:
            y_offset = 120
            for measure_name, data in measurements.items():
                if measure_name != 'unit':
                    text = f"{measure_name}: {data['value']} {measurements['unit']} ({data['confidence']}%)"
                    cv2.putText(frame, text, (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 25
        
        # Instructions
        if not self.measurements_locked:
            cv2.putText(frame, "Press SPACE to capture | ESC to exit", (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "Press 'R' to reset | ESC to exit", (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run_realtime(self, user_height: float, height_unit: str = 'cm', 
                    camera_index: int = 0):
        """
        Run real-time measurement system with webcam.
        
        Args:
            user_height: User's actual height
            height_unit: Unit of measurement (cm, inches, etc.)
            camera_index: Camera device index (default: 0)
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("="*60)
        print("VIRTUAL TAILORING SYSTEM - REAL-TIME MODE")
        print("="*60)
        print(f"User Height: {user_height} {height_unit}")
        print("\nInstructions:")
        print("1. Stand back so your full body is visible")
        print("2. Adopt a T-Pose (arms straight out, feet visible)")
        print("3. Press SPACE when ready to capture measurements")
        print("4. Press ESC to exit")
        print("="*60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect pose
            keypoints = self.detect_pose(frame)
            
            is_valid = False
            error_msg = "No person detected"
            current_measurements = None
            
            if keypoints is not None:
                # Draw skeleton
                frame = self.draw_skeleton(frame, keypoints)
                
                # Validate T-Pose
                is_valid, error_msg = self.validate_tpose(keypoints)
                
                if is_valid and not self.measurements_locked:
                    # Calculate measurements in real-time
                    current_measurements = self.calculate_measurements(
                        keypoints, user_height, height_unit
                    )
            
            # Use locked measurements if available
            display_measurements = self.final_measurements if self.measurements_locked else current_measurements
            
            # Draw overlay
            frame = self.draw_overlay(frame, is_valid, error_msg, display_measurements)
            
            # Display frame
            cv2.imshow('Virtual Tailoring - Real-Time', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32 and is_valid and not self.measurements_locked:  # SPACE
                self.measurements_locked = True
                self.final_measurements = current_measurements
                print("\n" + "="*60)
                print("MEASUREMENTS CAPTURED")
                print("="*60)
                for measure_name, data in self.final_measurements.items():
                    if measure_name != 'unit':
                        print(f"{measure_name:<20}: {data['value']} {self.final_measurements['unit']} "
                              f"(Confidence: {data['confidence']}%)")
                print("="*60)
            elif key == ord('r'):  # Reset
                self.measurements_locked = False
                self.final_measurements = None
                print("\nMeasurements reset. Adopt T-Pose to capture again.")
        
        cap.release()
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    # Initialize the system
    tailor = VirtualTailorMeasurements()
    
    # Get user input
    print("\n" + "="*60)
    print("VIRTUAL TAILORING MEASUREMENT SYSTEM")
    print("="*60)
    
    try:
        user_height = float(input("Enter your height: "))
        height_unit = input("Enter unit (cm/inches/m): ").strip().lower()
        
        if height_unit not in ['cm', 'inches', 'in', 'm']:
            print("Invalid unit. Using 'cm' by default.")
            height_unit = 'cm'
        
        if height_unit == 'in':
            height_unit = 'inches'
        
        # Run real-time measurement
        tailor.run_realtime(user_height, height_unit)
        
    except ValueError:
        print("Invalid input. Please enter a numeric value for height.")
    except KeyboardInterrupt:
        print("\nExiting...")