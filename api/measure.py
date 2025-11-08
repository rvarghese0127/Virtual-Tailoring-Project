from http.server import BaseHTTPRequestHandler
import json
import base64
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Tuple, Optional
import math
import io
from PIL import Image

class VirtualTailorMeasurements:
    """
    Computer Vision system for virtual tailoring measurements using T-Pose detection.
    Uses Ultralytics YOLO for pose estimation.
    """
    
    def __init__(self, model_name: str = 'yolov8n-pose.pt'):
        """Initialize the measurement system."""
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
    
    def detect_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect pose keypoints from image frame."""
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
        """Validate if the pose is a proper T-Pose."""
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
        """Calculate tailoring measurements from keypoints."""
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


# Global model instance (initialized once)
tailor = None

def get_tailor():
    """Get or create the tailor model instance."""
    global tailor
    if tailor is None:
        tailor = VirtualTailorMeasurements()
    return tailor


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        """Handle POST requests for image processing."""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse JSON
            data = json.loads(post_data.decode('utf-8'))
            
            # Get image data (base64 encoded)
            image_data = data.get('image', '')
            user_height = float(data.get('height', 170))
            height_unit = data.get('unit', 'cm')
            
            # Decode base64 image
            if image_data.startswith('data:image'):
                # Remove data URL prefix
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            
            # Convert to OpenCV format
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Could not decode image")
            
            # Get tailor instance
            tailor = get_tailor()
            
            # Detect pose
            keypoints = tailor.detect_pose(frame)
            
            if keypoints is None:
                response = {
                    'success': False,
                    'error': 'No person detected in image'
                }
            else:
                # Validate T-Pose
                is_valid, error_msg = tailor.validate_tpose(keypoints)
                
                if not is_valid:
                    response = {
                        'success': False,
                        'error': error_msg
                    }
                else:
                    # Calculate measurements
                    measurements = tailor.calculate_measurements(keypoints, user_height, height_unit)
                    response = {
                        'success': True,
                        'measurements': measurements
                    }
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            # Send error response
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            error_response = {
                'success': False,
                'error': str(e)
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

