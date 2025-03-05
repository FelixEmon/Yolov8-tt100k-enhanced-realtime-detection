import streamlit as st
import cv2
import PIL
from ultralytics import YOLO
import params

# Load YOLO model
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error("Unable to load the model.")
        st.error(e)
        return None

# Perform object detection on video
def detect_objects_in_video(conf, model):
    col1, col2 = st.columns(2)

    with col1:
        id_vid = st.sidebar.selectbox("Choose a video...", params.DICT_VID.keys())
        with open(params.DICT_VID.get(id_vid), 'rb') as video_file:
            bytes = video_file.read()

        if bytes:
            st.video(bytes)

    with col2:
        if st.sidebar.button('Start Detection'):
            try:
                capt = cv2.VideoCapture(str(params.DICT_VID.get(id_vid)))
                st_frame = st.empty()

                while capt.isOpened():
                    ret, img = capt.read()
                    if ret:
                        img = cv2.resize(img, (2560, 1440))
                        res = model.predict(img, conf=conf)
                        res_plot = res[0].plot()
                        st_frame.image(res_plot, caption='Detected Video', channels="BGR", use_container_width=True)
                    else:
                        capt.release()
                        break
            except Exception as e:
                st.sidebar.error("Error while loading video file: " + str(e))

# Perform object detection on image
def detect_objects_in_image(conf, model):
    src_img = st.sidebar.file_uploader("Upload an image...", type=("jpg", "png", "jpeg"))
    selected_img = st.sidebar.selectbox("... or choose an image.", params.DICT_IMG.keys())
    col1, col2 = st.columns(2)

    with col1:
        try:
            if src_img is None:
                uploaded_img = PIL.Image.open(params.DICT_IMG.get(selected_img))
                st.image(uploaded_img, caption="Chosen Image", use_container_width=True)
            else:
                uploaded_img = PIL.Image.open(src_img)
                st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error("Image is not uploaded.")
            st.error(e)

    with col2:
        if uploaded_img is not None:
            if st.sidebar.button('Start Detection'):
                res = model.predict(uploaded_img, conf=conf)
                boxes = res[0].boxes
                res_plot = res[0].plot()[:, :, ::-1]
                st.image(res_plot, caption='Detected Image', use_container_width=True)
        else:
            st.write("No image is uploaded.")
