import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp
import math

st.set_page_config(page_title="Live Face & Posture Analytics", layout="wide")
st.title("🖥️ Live Face, Gaze & Posture Dashboard")

# Sidebar controls
yaw_thresh = st.sidebar.slider("Yaw threshold (°) for ‘looking at camera’", 0.0, 45.0, 15.0)
center_offset = st.sidebar.slider("Max pixel offset for ‘centered face’", 0, 200, 80)

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # MediaPipe detectors
        self.mp_face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5
        )
        # 3D model points for solvePnP (nose, chin, eyes, mouth corners)
        self.model_points = np.array([
            [0.0,   0.0,    0.0],       # Nose tip
            [0.0,  -330.0, -65.0],      # Chin
            [-225.0, 170.0, -135.0],    # Left eye left corner
            [225.0, 170.0, -135.0],     # Right eye right corner
            [-150.0,-150.0,-125.0],     # Mouth left corner
            [150.0,-150.0,-125.0]       # Mouth right corner
        ], dtype="double")

    def estimate_head_pose(self, img, landmarks):
        h, w, _ = img.shape
        # 2D image points from MediaPipe
        image_points = np.array([
            [landmarks[1].x * w,   landmarks[1].y * h],   # Nose tip
            [landmarks[152].x * w, landmarks[152].y * h], # Chin
            [landmarks[33].x * w,  landmarks[33].y * h],  # Left eye corner
            [landmarks[263].x * w, landmarks[263].y * h], # Right eye corner
            [landmarks[61].x * w,  landmarks[61].y * h],  # Mouth left
            [landmarks[291].x * w, landmarks[291].y * h], # Mouth right
        ], dtype="double")

        # Camera matrix ≈ focal length = width
        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4,1))  # no lens distortion

        # Solve PnP
        success, rotation_vec, _ = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        # Convert to Euler angles (in degrees)
        rot_mat, _ = cv2.Rodrigues(rotation_vec)
        proj_mat = np.hstack((rot_mat, np.zeros((3,1))))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
        yaw = float(euler[1])  # yaw angle
        return yaw

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # Face detection
        det_results = self.mp_face_det.process(rgb)
        if det_results.detections:
            det = det_results.detections[0]
            bbox = det.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = x1 + int(bbox.width * w)
            y2 = y1 + int(bbox.height * h)
            # Draw box
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

            # Face center offset
            face_cx, face_cy = (x1+x2)//2, (y1+y2)//2
            frame_cx, frame_cy = w//2, h//2
            dx, dy = face_cx-frame_cx, face_cy-frame_cy

            # Gaze / head pose
            mesh_results = self.mp_face_mesh.process(rgb)
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0].landmark
                yaw = self.estimate_head_pose(img, landmarks)

                looking = abs(yaw) <= yaw_thresh
                centered = abs(dx) <= center_offset and abs(dy) <= center_offset

                # Overlay text
                txt = f"Yaw: {yaw:.1f}° → {'LOOKING' if looking else 'AWAY'}"
                cv2.putText(img, txt, (x1, y1-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0,255,255) if looking else (0,0,255), 2)
                pos_txt = f"Offset: [{dx},{dy}] → {'CENTERED' if centered else 'OFF-CENTER'}"
                cv2.putText(img, pos_txt, (x1, y1-50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255,255,0) if centered else (0,0,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="live-face",
    mode="SENDRECV",
    rtc_configuration=RTC_CONFIGURATION,
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={
        "video": {"width": 640, "height": 480, "frameRate": 30}
    },
)
