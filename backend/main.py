from flask import Flask, Response
from flask_cors import CORS
import cv2
from yoga_trainer import yoga_trainer_bp, detect_pose, pose_video

app = Flask(__name__)
# Allow all origins for development (restrict in production)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process frame if pose_video is initialized
            if pose_video:
                frame, landmarks = detect_pose(frame, pose_video)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Register Blueprints
app.register_blueprint(yoga_trainer_bp, url_prefix='/yoga')

if __name__ == '__main__':
    try:
        port = 5000
        print(f"Server running on port: http://localhost:{port}")
        app.run(debug=True, port=port)
    except Exception as e:
        print(f"Error starting server on port 5000: {e}")
        port = 5500
        print(f"Server running on port: http://localhost:{port}")
        app.run(debug=True, port=port)