"""
Facial landmark detection and utilities.
Provides functions for detecting and working with facial landmarks.
"""

import cv2
import numpy as np
import dlib

def get_landmarks(image, face_detector=None, landmark_predictor=None):
    """
    Enhanced face detection with better fallbacks and diagnostics.
    
    Args:
        image: Input image
        face_detector: Optional dlib face detector (if None, will try to create one)
        landmark_predictor: Optional dlib landmark predictor
        
    Returns:
        list: List of (x, y) landmark coordinates, or None if detection failed
    """
    # Check if we need to initialize detectors
    if face_detector is None or landmark_predictor is None:
        try:
            # Try to initialize detectors
            if face_detector is None:
                face_detector = dlib.get_frontal_face_detector()
                
            if landmark_predictor is None:
                print("No landmark predictor provided - landmarks will be estimated geometrically")
        except Exception as e:
            print(f"Error initializing face detection: {str(e)}")
            return create_geometric_landmarks(image)
    
    # Check if we have both required components
    if face_detector is None:
        print("Face detector not available")
        return create_geometric_landmarks(image)
            
    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Print some diagnostics about the image
    print(f"Image shape for face detection: {image.shape}")
    print(f"Image intensity range: {np.min(gray)} to {np.max(gray)}")
    
    # Try detection with different upsample factors
    for upsample_factor in [0, 1, 2]:  # Try no upsampling, then 1x, then 2x
        print(f"Attempting face detection with upsample factor {upsample_factor}")
        faces = face_detector(gray, upsample_factor)
        
        if faces:
            print(f"Detected {len(faces)} faces with upsample factor {upsample_factor}")
            # Use largest face by area
            largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Get landmarks if predictor is available
            if landmark_predictor is not None:
                try:
                    shape = landmark_predictor(gray, largest_face)
                    landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
                    print(f"Successfully extracted {len(landmarks)} landmarks")
                    return landmarks
                except Exception as e:
                    print(f"Error detecting landmarks: {str(e)}")
                    # Try next upsample factor
                    continue
            else:
                # If no predictor, estimate based on face rectangle
                x, y, w, h = (largest_face.left(), largest_face.top(), 
                              largest_face.width(), largest_face.height())
                return estimate_landmarks_from_face_rect(x, y, w, h, image.shape[1], image.shape[0])
    
    # If we reach here, no detection succeeded
    print("No faces detected with any upsampling, using geometric fallback")
    return create_geometric_landmarks(image)

def create_geometric_landmarks(image):
    """
    Create a full set of 68 estimated facial landmarks based on image geometry when detection fails.
    
    Args:
        image: Input image
        
    Returns:
        list: List of estimated landmark positions
    """
    h, w = image.shape[:2]
    
    landmarks = []
    
    # Jaw line (points 0-16)
    for i in range(17):
        angle = (i / 16.0) * np.pi
        radius = min(w, h) * 0.45
        x = int(w/2 - np.cos(angle) * radius)
        y = int(h/2 + np.sin(angle) * radius)
        landmarks.append((x, y))
    
    # Eyebrows (points 17-26)
    eyebrow_y = int(h * 0.3)
    for i in range(5):  # Left eyebrow
        x = int(w * (0.2 + i * 0.06))
        y = eyebrow_y
        landmarks.append((x, y))
    
    for i in range(5):  # Right eyebrow
        x = int(w * (0.8 - i * 0.06))
        y = eyebrow_y
        landmarks.append((x, y))
    
    # Nose (points 27-35)
    nose_bridge_top_y = int(h * 0.35)
    nose_tip_y = int(h * 0.5)
    nose_width = int(w * 0.1)
    
    # Nose bridge (vertical line from between eyebrows to tip)
    for i in range(4):
        x = int(w / 2)
        y = int(nose_bridge_top_y + (nose_tip_y - nose_bridge_top_y) * (i / 3.0))
        landmarks.append((x, y))
    
    # Nose base (horizontal line below tip)
    for i in range(5):
        x = int(w/2 + nose_width * (-1 + i/2.0))
        y = nose_tip_y + int(h * 0.02)
        landmarks.append((x, y))
    
    # Eyes (points 36-47)
    eye_y = int(h * 0.38)
    left_eye_x = int(w * 0.3)
    right_eye_x = int(w * 0.7)
    eye_width = int(w * 0.07)
    
    # Left eye (6 points)
    for i in range(6):
        angle = (i / 6.0) * 2 * np.pi
        x = int(left_eye_x + np.cos(angle) * eye_width)
        y = int(eye_y + np.sin(angle) * (eye_width * 0.4))
        landmarks.append((x, y))
    
    # Right eye (6 points)
    for i in range(6):
        angle = (i / 6.0) * 2 * np.pi
        x = int(right_eye_x + np.cos(angle) * eye_width)
        y = int(eye_y + np.sin(angle) * (eye_width * 0.4))
        landmarks.append((x, y))
    
    # Outer lips (points 48-59)
    mouth_y = int(h * 0.7)
    mouth_width = int(w * 0.3)
    mouth_height = int(h * 0.06)
    
    for i in range(12):
        angle = (i / 12.0) * 2 * np.pi
        x = int(w/2 + np.cos(angle) * mouth_width/2)
        y = int(mouth_y + np.sin(angle) * mouth_height/2)
        landmarks.append((x, y))
    
    # Inner lips (points 60-67)
    inner_mouth_width = mouth_width * 0.7
    inner_mouth_height = mouth_height * 0.7
    
    for i in range(8):
        angle = (i / 8.0) * 2 * np.pi
        x = int(w/2 + np.cos(angle) * inner_mouth_width/2)
        y = int(mouth_y + np.sin(angle) * inner_mouth_height/2)
        landmarks.append((x, y))
    
    print("Using fallback geometric landmarks")
    return landmarks

def estimate_landmarks_from_face_rect(x, y, w, h, img_width, img_height):
    """
    Estimate facial landmarks based on detected face rectangle.
    More accurate than pure geometric estimation but less accurate than real detection.
    
    Args:
        x, y, w, h: Face rectangle coordinates and dimensions
        img_width, img_height: Image dimensions
        
    Returns:
        list: List of estimated landmark positions
    """
    landmarks = []
    
    # Face center
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    # Jaw line (points 0-16)
    jaw_y = y + int(h * 0.8)
    jaw_width = w
    
    for i in range(17):
        ratio = i / 16.0
        jaw_x = x + int(jaw_width * ratio)
        
        # Add some curvature to the jawline
        offset_y = int(h * 0.05 * np.sin(np.pi * ratio))
        landmarks.append((jaw_x, jaw_y - offset_y))
    
    # Eyebrows (points 17-26)
    eyebrow_y = y + int(h * 0.25)
    
    # Left eyebrow
    left_eyebrow_start_x = x + int(w * 0.15)
    left_eyebrow_end_x = x + int(w * 0.45)
    
    for i in range(5):
        ratio = i / 4.0
        eyebrow_x = left_eyebrow_start_x + int((left_eyebrow_end_x - left_eyebrow_start_x) * ratio)
        
        # Add some arch to the eyebrow
        offset_y = int(h * 0.02 * np.sin(np.pi * ratio))
        landmarks.append((eyebrow_x, eyebrow_y - offset_y))
    
    # Right eyebrow
    right_eyebrow_start_x = x + int(w * 0.55)
    right_eyebrow_end_x = x + int(w * 0.85)
    
    for i in range(5):
        ratio = i / 4.0
        eyebrow_x = right_eyebrow_end_x - int((right_eyebrow_end_x - right_eyebrow_start_x) * ratio)
        
        # Add some arch to the eyebrow
        offset_y = int(h * 0.02 * np.sin(np.pi * ratio))
        landmarks.append((eyebrow_x, eyebrow_y - offset_y))
    
    # Nose (points 27-35)
    nose_bridge_top_y = y + int(h * 0.3)
    nose_tip_y = y + int(h * 0.55)
    nose_width = int(w * 0.2)
    
    # Nose bridge
    for i in range(4):
        ratio = i / 3.0
        nose_y = nose_bridge_top_y + int((nose_tip_y - nose_bridge_top_y) * ratio)
        landmarks.append((face_center_x, nose_y))
    
    # Nose base
    nose_base_y = nose_tip_y + int(h * 0.03)
    for i in range(5):
        ratio = (i - 2) / 2.0  # -1 to 1
        nose_x = face_center_x + int(nose_width * ratio)
        landmarks.append((nose_x, nose_base_y))
    
    # Eyes (points 36-47)
    eye_y = y + int(h * 0.35)
    eye_width = int(w * 0.15)
    eye_height = int(h * 0.06)
    
    # Left eye center
    left_eye_center_x = x + int(w * 0.3)
    
    # Left eye (6 points)
    for i in range(6):
        angle = (i / 6.0) * 2 * np.pi
        eye_x = int(left_eye_center_x + np.cos(angle) * eye_width / 2)
        eye_y_offset = int(np.sin(angle) * eye_height / 2)
        landmarks.append((eye_x, eye_y + eye_y_offset))
    
    # Right eye center
    right_eye_center_x = x + int(w * 0.7)
    
    # Right eye (6 points)
    for i in range(6):
        angle = (i / 6.0) * 2 * np.pi
        eye_x = int(right_eye_center_x + np.cos(angle) * eye_width / 2)
        eye_y_offset = int(np.sin(angle) * eye_height / 2)
        landmarks.append((eye_x, eye_y + eye_y_offset))
    
    # Mouth (points 48-67)
    mouth_y = y + int(h * 0.75)
    mouth_width = int(w * 0.4)
    mouth_height = int(h * 0.1)
    
    # Outer mouth (12 points)
    for i in range(12):
        angle = (i / 12.0) * 2 * np.pi
        mouth_x = int(face_center_x + np.cos(angle) * mouth_width / 2)
        mouth_y_offset = int(np.sin(angle) * mouth_height / 2)
        landmarks.append((mouth_x, mouth_y + mouth_y_offset))
    
    # Inner mouth (8 points)
    inner_mouth_width = mouth_width * 0.7
    inner_mouth_height = mouth_height * 0.7
    
    for i in range(8):
        angle = (i / 8.0) * 2 * np.pi
        mouth_x = int(face_center_x + np.cos(angle) * inner_mouth_width / 2)
        mouth_y_offset = int(np.sin(angle) * inner_mouth_height / 2)
        landmarks.append((mouth_x, mouth_y + mouth_y_offset))
    
    return landmarks

def estimate_bangs_position(landmarks, height, width):
    """
    Estimates where bangs should be positioned based on facial landmarks.
    
    Args:
        landmarks: Facial landmarks from the source image
        height: Height of the image
        width: Width of the image
        
    Returns:
        dict: Contains estimated position information
    """
    if landmarks is None or len(landmarks) < 17:
        # If no landmarks, use geometric center and top 20%
        return {
            'center_x': width // 2,
            'top_y': int(height * 0.2),
            'width': int(width * 0.4),
            'height': int(height * 0.15)
        }
    
    try:
        # Get eyebrow positions
        if len(landmarks) >= 27:
            eyebrow_points = landmarks[17:27]  # Standard eyebrow landmarks
            eyebrow_y = min([p[1] for p in eyebrow_points])
            left_x = min([p[0] for p in eyebrow_points])
            right_x = max([p[0] for p in eyebrow_points])
        else:
            # Fallback if eyebrow landmarks aren't available
            # Use top of face
            top_points = [landmarks[i] for i in range(min(8, len(landmarks)))]
            eyebrow_y = min([p[1] for p in top_points]) + int(height * 0.1)
            left_x = min([p[0] for p in top_points])
            right_x = max([p[0] for p in top_points]) 
        
        # Calculate bangs parameters
        center_x = (left_x + right_x) // 2
        bangs_width = int((right_x - left_x) * 1.2)  # Make slightly wider than eyebrows
        top_y = max(0, eyebrow_y - int(height * 0.15))  # Place above eyebrows
        bangs_height = eyebrow_y - top_y
        
        return {
            'center_x': center_x,
            'top_y': top_y,
            'width': bangs_width,
            'height': bangs_height,
            'eyebrow_y': eyebrow_y,
            'left_x': left_x,
            'right_x': right_x
        }
    
    except Exception as e:
            print(f"Error estimating bangs position: {str(e)}")
            # Fallback to simple geometric positioning
            return {
                'center_x': width // 2,
                'top_y': int(height * 0.2),
                'width': int(width * 0.4),
                'height': int(height * 0.15)
            }

def analyze_face_proportions(landmarks):
    """
    Analyze facial proportions from landmarks to help with bangs positioning.
    
    Args:
        landmarks: Facial landmarks
        
    Returns:
        dict: Dictionary of facial measurements and proportions
    """
    if landmarks is None or len(landmarks) < 68:
        return None
        
    try:
        # Get key points
        # Face height (chin to hairline)
        face_top = min([p[1] for p in landmarks[:17]])  # Top of face contour
        chin_y = landmarks[8][1]  # Chin point
        face_height = chin_y - face_top
        
        # Face width (at cheekbones)
        face_left = min([p[0] for p in landmarks[:17]])
        face_right = max([p[0] for p in landmarks[:17]])
        face_width = face_right - face_left
        
        # Eyes
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        eye_distance = right_eye_center[0] - left_eye_center[0]
        
        # Eyebrows
        left_eyebrow_y = min([p[1] for p in landmarks[17:22]])
        right_eyebrow_y = min([p[1] for p in landmarks[22:27]])
        eyebrow_y = min(left_eyebrow_y, right_eyebrow_y)
        
        # Forehead height (eyebrows to top of face)
        forehead_height = eyebrow_y - face_top
        
        # Calculate proportions
        forehead_ratio = forehead_height / face_height if face_height > 0 else 0.33
        
        return {
            'face_height': face_height,
            'face_width': face_width,
            'eye_distance': eye_distance,
            'forehead_height': forehead_height,
            'forehead_ratio': forehead_ratio,
            'eyebrow_y': eyebrow_y,
            'face_top_y': face_top
        }
    except Exception as e:
        print(f"Error analyzing face proportions: {str(e)}")
        return None