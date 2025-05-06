from flask import Blueprint, jsonify
import math
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import threading

# Initialize Blueprint
yoga_trainer_bp = Blueprint('yoga_trainer', __name__)

# Initialize mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global variables
yoga_status = {
    "completed": False,
    "current_pose": None,
    "progress": 0,
    "detected_pose": None,
    "accuracy": 0
}

pose_array = ['T Pose', 'Tree Pose', 'Warrior II Pose']
current_pose_index = 0
pose_detected_count = 0
pose_video = None
camera_video = None
processing_active = False

# Initialize text-to-speech
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

@yoga_trainer_bp.route('/yoga_trainer')
def yoga_trainer_home():
    return "Yoga Trainer Home"

@yoga_trainer_bp.route('/start-yoga', methods=['POST'])
def start_yoga():
    global processing_active, current_pose_index, pose_detected_count, pose_video, camera_video
    
    try:
        # Initialize variables
        yoga_status.update({
            "completed": False,
            "current_pose": pose_array[0],
            "progress": 0,
            "detected_pose": None,
            "accuracy": 0
        })
        
        current_pose_index = 0
        pose_detected_count = 0
        
        # Initialize pose detection
        pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=1)
        
        # Start processing thread
        processing_active = True
        thread = threading.Thread(target=process_poses)
        thread.daemon = True
        thread.start()
        
        speak(f"Starting yoga session. First pose: {pose_array[0]}")

        return jsonify({"message": "Yoga session started!", "status": yoga_status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@yoga_trainer_bp.route('/stop-yoga', methods=['POST'])
def stop_yoga():
    global processing_active, pose_video, camera_video
    
    try:
        processing_active = False
        if pose_video:
            pose_video.close()
        if camera_video and camera_video.isOpened():
            camera_video.release()
            
        yoga_status.update({
            "completed": False,
            "current_pose": None,
            "progress": 0,
            "detected_pose": None,
            "accuracy": 0
        })
        
        return jsonify({"message": "Yoga session stopped!", "status": yoga_status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@yoga_trainer_bp.route('/yoga-status')
def get_yoga_status():
    return jsonify(yoga_status)

def process_poses():
    global current_pose_index, pose_detected_count, processing_active, yoga_status
    
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 960)
    
    while processing_active and current_pose_index < len(pose_array):
        ret, frame = camera_video.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        
        frame, landmarks = detect_pose(frame, pose_video)
        
        if landmarks:
            frame, label_detected = classify_pose(landmarks, frame)
            yoga_status['detected_pose'] = label_detected
            
            if label_detected == pose_array[current_pose_index]:
                pose_detected_count += 1
                yoga_status['progress'] = min(100, (pose_detected_count / 30) * 100)
                yoga_status['accuracy'] = calculate_accuracy(landmarks, pose_array[current_pose_index])
                
                if pose_detected_count >= 30:
                    pose_detected_count = 0
                    current_pose_index += 1
                    
                    if current_pose_index < len(pose_array):
                        yoga_status['current_pose'] = pose_array[current_pose_index]
                        yoga_status['progress'] = 0
                        speak(f"Next pose: {pose_array[current_pose_index]}")
                    else:
                        yoga_status['completed'] = True
                        speak("Congratulations! You've completed all poses!")
                        break
    
    if camera_video.isOpened():
        camera_video.release()

def detect_pose(image, pose, display=False):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                 connections=mp_pose.POSE_CONNECTIONS)
        landmarks = [(int(landmark.x * width), int(landmark.y * height), landmark.z * width)
                     for landmark in results.pose_landmarks.landmark]

    return output_image, landmarks

def calculate_angle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def calculate_accuracy(landmarks, target_pose):
    # Simplified accuracy calculation - adjust as needed
    try:
        if target_pose == 'T Pose':
            left_shoulder_angle = calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            )
            right_shoulder_angle = calculate_angle(
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            )
            # Ideal angles for T-pose
            ideal_angle = 90
            accuracy = 100 - (abs(left_shoulder_angle - ideal_angle) + abs(right_shoulder_angle - ideal_angle)) / 2
            return max(0, min(100, accuracy))
        
        elif target_pose == 'Tree Pose':
            # Add specific calculations for Tree Pose
            return 80  # Placeholder
            
        elif target_pose == 'Warrior II Pose':
            # Add specific calculations for Warrior II Pose
            return 80  # Placeholder
            
    except:
        return 0

def classify_pose(landmarks, output_image, display=False):
    label = 'Unknown Pose'
    color = (0, 0, 255)

    try:
        left_elbow_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        )

        right_elbow_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        )

        left_shoulder_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        )

        right_shoulder_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        )

        left_knee_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )

        right_knee_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )

        # T Pose detection
        if (left_elbow_angle > 165 and left_elbow_angle < 195 and 
            right_elbow_angle > 165 and right_elbow_angle < 195 and
            left_shoulder_angle > 80 and left_shoulder_angle < 110 and 
            right_shoulder_angle > 80 and right_shoulder_angle < 110 and
            left_knee_angle > 160 and left_knee_angle < 195 and 
            right_knee_angle > 160 and right_knee_angle < 195):
            label = 'T Pose'

        # Tree Pose detection
        elif ((left_knee_angle > 315 and left_knee_angle < 335) or 
              (right_knee_angle > 25 and right_knee_angle < 45)):
            label = 'Tree Pose'

        # Warrior II Pose detection
        elif (left_elbow_angle > 165 and left_elbow_angle < 195 and 
              right_elbow_angle > 165 and right_elbow_angle < 195 and
              left_shoulder_angle > 80 and left_shoulder_angle < 110 and 
              right_shoulder_angle > 80 and right_shoulder_angle < 110 and
              ((left_knee_angle > 90 and left_knee_angle < 120) or 
               (right_knee_angle > 90 and right_knee_angle < 120))):
            label = 'Warrior II Pose'

        if label != 'Unknown Pose':
            color = (0, 255, 0)

        cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        return output_image, label

    except Exception as e:
        print(f"Error classifying pose: {e}")
        return output_image, label


# Js

#         // Load the TensorFlow model
#         // async function loadModel() {
#         //     try {
#         //         await tf.ready();
#         //         const model = poseDetection.SupportedModels.MoveNet;
#         //         const detectorConfig = {
#         //             modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
#         //             enableSmoothing: true
#         //         };
#         //         detector = await poseDetection.createDetector(model, detectorConfig);
#         //         modelLoaded = true;
#         //         console.log("Model loaded successfully");
#         //     } catch (error) {
#         //         console.error("Error loading model:", error);
#         //         poseFeedback.textContent = "Error loading pose detection. Please refresh the page.";
#         //     }
#         // }



#         // Detect pose
#         // async function detectPose(video) {
#         //     try {
#         //         poses = await detector.estimatePoses(video);
#         //         drawPoses(video);
                
#         //         // Check if pose is correct
#         //         checkPose(poses);
                
#         //        poseFeedback.textContent = "Error detecting pose. Please try again.";
#         //           animationId = requestAnimationFrame(() => detectPose(video));
#         //     } catch (error) {
#         //         console.error("Error detecting pose:", error);
#         //        cancelAnimationFrame(animationId);
#         //     }
#         // }

#         // Draw poses on canvas
#         // function drawPoses(video) {
#         //     ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
#         //     ctx.drawImage(video, 0, 0, outputCanvas.width, outputCanvas.height);
            
#         //     if (poses && poses.length > 0) {
#         //         drawKeypoints(poses[0].keypoints);
#         //         drawSkeleton(poses[0].keypoints);
#         //     }
            
#         //     // Update video element with processed frame
#         //     videoElement.srcObject = null;
#         //     videoElement.poster = outputCanvas.toDataURL();
#         // }

#         // Draw keypoints
#         // function drawKeypoints(keypoints) {
#         //     const keypointIndices = {
#         //         'nose': 0,
#         //         'left_eye': 1,
#         //         'right_eye': 2,
#         //         'left_ear': 3,
#         //         'right_ear': 4,
#         //         'left_shoulder': 5,
#         //         'right_shoulder': 6,
#         //         'left_elbow': 7,
#         //         'right_elbow': 8,
#         //         'left_wrist': 9,
#         //         'right_wrist': 10,
#         //         'left_hip': 11,
#         //         'right_hip': 12,
#         //         'left_knee': 13,
#         //         'right_knee': 14,
#         //         'left_ankle': 15,
#         //         'right_ankle': 16
#         //     };

#         //     for (let i = 0; i < keypoints.length; i++) {
#         //         const keypoint = keypoints[i];
#         //         if (keypoint.score > 0.3) {
#         //             const { y, x } = keypoint;
#         //             ctx.beginPath();
#         //             ctx.arc(x, y, 5, 0, 2 * Math.PI);
#         //             ctx.fillStyle = 'red';
#         //             ctx.fill();
#         //         }
#         //     }
#         // }

#         // // Draw skeleton
#         // function drawSkeleton(keypoints) {
#         //     const adjacentKeyPoints = [
#         //         [5, 6],   // shoulders
#         //         [5, 7],   // left shoulder to left elbow
#         //         [6, 8],   // right shoulder to right elbow
#         //         [7, 9],   // left elbow to left wrist
#         //         [8, 10],  // right elbow to right wrist
#         //         [5, 11],  // left shoulder to left hip
#         //         [6, 12],  // right shoulder to right hip
#         //         [11, 12], // hips
#         //         [11, 13], // left hip to left knee
#         //         [12, 14], // right hip to right knee
#         //         [13, 15], // left knee to left ankle
#         //         [14, 16]  // right knee to right ankle
#         //     ];

#         //     ctx.strokeStyle = '#00FF00';
#         //     ctx.lineWidth = 2;

#         //     adjacentKeyPoints.forEach(([i, j]) => {
#         //         const kp1 = keypoints[i];
#         //         const kp2 = keypoints[j];
                
#         //         if (kp1.score > 0.3 && kp2.score > 0.3) {
#         //             ctx.beginPath();
#         //             ctx.moveTo(kp1.x, kp1.y);
#         //             ctx.lineTo(kp2.x, kp2.y);
#         //             ctx.stroke();
#         //         }
#         //     });
#         // }

#         // // Check if pose is correct
#         // function checkPose(poses) {
#         //     if (poses.length === 0) return;
            
#         //     const keypoints = poses[0].keypoints;
#         //     let isCorrect = false;
            
#         //     switch(currentPose) {
#         //         case 0: // T Pose
#         //             isCorrect = checkTPose(keypoints);
#         //             break;
#         //         case 1: // Tree Pose
#         //             isCorrect = checkTreePose(keypoints);
#         //             break;
#         //         case 2: // Warrior Pose
#         //             isCorrect = checkWarriorPose(keypoints);
#         //             break;
#         //     }
            
#         //     if (isCorrect) {
#         //         poseFeedback.textContent = `Great job! Hold the ${poseNames[currentPose]} for ${countdown} more seconds`;
#         //         poseFeedback.style.color = "green";
#         //     } else {
#         //         poseFeedback.textContent = `Adjust your position to match the ${poseNames[currentPose]}`;
#         //         poseFeedback.style.color = "red";
#         //     }
#         // }

#         // // Check T Pose
#         // function checkTPose(keypoints) {
#         //     // Simple check - shoulders and wrists should be aligned horizontally
#         //     const leftShoulder = keypoints[5];  // left_shoulder
#         //     const rightShoulder = keypoints[6]; // right_shoulder
#         //     const leftWrist = keypoints[9];     // left_wrist
#         //     const rightWrist = keypoints[10];   // right_wrist
            
#         //     if (!leftShoulder || !rightShoulder || !leftWrist || !rightWrist) return false;
            
#         //     // Check if wrists are at similar height to shoulders
#         //     const shoulderHeight = (leftShoulder.y + rightShoulder.y) / 2;
#         //     const wristHeight = (leftWrist.y + rightWrist.y) / 2;
            
#         //     return Math.abs(shoulderHeight - wristHeight) < 30;
#         // }

#         // // Check Tree Pose
#         // function checkTreePose(keypoints) {
#         //     // Simple check - one foot should be near the other knee
#         //     const leftAnkle = keypoints[15];  // left_ankle
#         //     const rightAnkle = keypoints[16]; // right_ankle
#         //     const leftKnee = keypoints[13];   // left_knee
#         //     const rightKnee = keypoints[14];  // right_knee
            
#         //     if (!leftAnkle || !rightAnkle || !leftKnee || !rightKnee) return false;
            
#         //     // Check if one ankle is near the opposite knee
#         //     const leftAnkleNearRightKnee = (
#         //         Math.abs(leftAnkle.x - rightKnee.x) < 30 && 
#         //         Math.abs(leftAnkle.y - rightKnee.y) < 30
#         //     );
            
#         //     const rightAnkleNearLeftKnee = (
#         //         Math.abs(rightAnkle.x - leftKnee.x) < 30 && 
#         //         Math.abs(rightAnkle.y - leftKnee.y) < 30
#         //     );
            
#         //     return leftAnkleNearRightKnee || rightAnkleNearLeftKnee;
#         // }

#         // // Check Warrior Pose
#         // function checkWarriorPose(keypoints) {
#         //     // Simple check - one knee should be bent significantly
#         //     const leftKnee = keypoints[13];  // left_knee
#         //     const rightKnee = keypoints[14]; // right_knee
#         //     const leftHip = keypoints[11];   // left_hip
#         //     const rightHip = keypoints[12];  // right_hip
            
#         //     if (!leftKnee || !rightKnee || !leftHip || !rightHip) return false;
            
#         //     // Check if one knee is significantly lower than the hip
#         //     const leftKneeBent = leftKnee.y > leftHip.y + 30;
#         //     const rightKneeBent = rightKnee.y > rightHip.y + 30;
            
#         //     return leftKneeBent || rightKneeBent;
#         // }
