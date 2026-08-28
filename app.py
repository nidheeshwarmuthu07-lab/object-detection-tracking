import av
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
)


# ---------------------------------------------------------
# PAGE
# ---------------------------------------------------------

st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Real-Time Object Detection & Tracking")

st.write(
    "YOLO is used for object detection and ByteTrack is used "
    "to track detected objects with unique IDs."
)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
)


# ---------------------------------------------------------
# VIDEO PROCESSOR
# ---------------------------------------------------------

class YOLOProcessor:

    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def recv(self, frame):

        # Convert browser frame to OpenCV/Numpy format
        image = frame.to_ndarray(format="bgr24")

        # YOLO detection + ByteTrack tracking
        results = self.model.track(
            source=image,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            imgsz=640,
            verbose=False,
        )

        # Get first result
        result = results[0]

        # Let Ultralytics draw:
        # bounding boxes
        # labels
        # confidence values
        # tracking IDs
        annotated_frame = result.plot()

        # Return processed frame to browser
        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )


# ---------------------------------------------------------
# CAMERA
# ---------------------------------------------------------

st.subheader("Live Camera")

st.info(
    "Click START, allow camera access, and wait a few seconds "
    "for the YOLO model to begin processing."
)


RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


webrtc_streamer(
    key="yolo-tracking",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)


# ---------------------------------------------------------
# INFO
# ---------------------------------------------------------

st.divider()

st.subheader("How it works")

st.markdown(
    """
    1. Browser captures the webcam video.
    2. YOLO detects objects in each frame.
    3. ByteTrack tracks detected objects.
    4. Each tracked object receives an ID.
    5. Bounding boxes and labels are displayed live.
    """
)

st.subheader("Technologies")

st.markdown(
    """
    - Python
    - YOLO11
    - Ultralytics
    - ByteTrack
    - Streamlit
    - WebRTC
    """
)
