import cv2
import streamlit as st
import numpy as np
import time
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

def detect_faces(color, min_neighbors, scale_factor):
    cap = cv2.VideoCapture(0)
    st_frame = st.empty()
    stop_button = st.button("Arreter Detection")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Convert to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

        # Check for stop every 100ms
        time.sleep(0.1)
        if stop_button:
            break

    cap.release()
    return frame_rgb if 'frame_rgb' in locals() else None

def app():
    st.title("Face Detection using Viola-Jones Algorithm")

    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Ajuster les parametres en dessous
        2. Cliquez 'Detecter visages' pour commencer
        3. Cliquez 'Arreter Detection' pour arreter
        4. Cliquez 'Sauvegarder Image' pour sauvegarder l'image
        """)

        st.subheader("Parameters")
        rect_color = st.color_picker("Rectangle Color", "#00FF00")
        min_neighbors = st.slider("minNeighbors", 1, 15, 5)
        scale_factor = st.slider("scaleFactor", 1.01, 2.0, 1.3, 0.01)

    if st.button("Detecter visages"):
        color = hex_to_bgr(rect_color)
        with st.spinner("Detecting faces..."):
            final_frame = detect_faces(color, min_neighbors, scale_factor)
            if final_frame is not None:
                st.session_state.last_frame = final_frame

    if 'last_frame' in st.session_state:
        st.image(st.session_state.last_frame, caption="Last Detected Frame", use_container_width =True)

        if st.button("Savuvergarder Image"):
            img = Image.fromarray(st.session_state.last_frame)
            img.save("detected_faces.png")
            st.success("Image saved as detected_faces.png")
            with open("detected_faces.png", "rb") as file:
                st.download_button(
                    label="Download Image",
                    data=file,
                    file_name="detected_faces.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    app()