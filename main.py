import streamlit as st
import params
import helper_funcs
from pathlib import Path


logo_path = "assets\logo\logo3.png"

def set_page_config():
    st.set_page_config(
        page_title="Traffic Signs Detection with YOLOv8",
        layout="wide",
        initial_sidebar_state="expanded",

    )

def main():
    set_page_config()
    
    st.title("Traffic Signs Detection with YOLOv8")
    st.write("Felix 2030026040")
    st.write("Jimmy 2030026220")
    st.write("Vincent 2030026009")
    st.markdown("---")
    custom_style = """
                <style>
                body {
                    background-color:white;
                }
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(custom_style, unsafe_allow_html=True)
   

    with st.sidebar:
        st.image(logo_path, use_container_width=True)
        st.header("Adjust confidence level here")
        conf = float(st.slider("Confidence Level", 0, 100, 40)) / 100
        st.markdown("---")

    model_path = Path(params.MODEL_DIR)

    try:
        model = helper_funcs.load_model(model_path)
    except Exception as e:
        st.error("Unable to load model.")
        st.error(e)

    st.sidebar.header("Image/Video Configurations")
    rb_source = st.sidebar.radio("Select Source", params.SOURCES_LIST)
    source_img = None

    if rb_source == params.IMAGE:
        helper_funcs.detect_objects_in_image(conf, model)

    elif rb_source == params.VIDEO:
        helper_funcs.detect_objects_in_video(conf, model)

    else:
        st.error("Please select a valid source!")
    

if __name__ == '__main__':
    main()
