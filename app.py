import cv2
import av
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Real-Time Object Detection & Tracking")

st.write(
    """
    This application uses YOLO11 for object detection
    and ByteTrack for object tracking.
    """
)

st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    0.10,
    1.00,
    0.40,
    0.05
)


class YOLOVideoProcessor:

    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        results = self.model.track(
            image,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            verbose=False
        )

        output_frame = image.copy()

        if results and len(results) > 0:

            result = results[0]

            if result.boxes is not None:

                for box in result.boxes:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]

                    conf = float(box.conf[0])

                    track_id = None

                    if box.id is not None:
                        track_id = int(box.id[0])

                    if track_id is not None:
                        label = f"{class_name} ID:{track_id} {conf:.2f}"
                    else:
                        label = f"{class_name} {conf:.2f}"

                    cv2.rectangle(
                        output_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        output_frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        return av.VideoFrame.from_ndarray(
            output_frame,
            format="bgr24"
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


st.subheader("Live Camera")

st.info("Click START and allow camera permission.")


webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


st.divider()

st.subheader("Technologies Used")

st.markdown(
    """
    - Python
    - YOLO11
    - Ultralytics
    - OpenCV
    - ByteTrack
    - Streamlit
    - WebRTC
    """
)
