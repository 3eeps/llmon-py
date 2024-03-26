# ./codespace/pages/background removal.py
import streamlit as st

st.set_page_config(page_title="background removal", page_icon="🍋", layout="wide", initial_sidebar_state="auto")
st.title("🍋background removal")

from llmonpy import llmonaid
from PIL import Image
from transformers import pipeline
import os

if st.session_state['approved_login'] == True:
    filename = "bkgd_upload_image.png"
    with st.sidebar:
        llmonaid.memory_display()

        rem_bak_file = st.file_uploader(label=":orange[remove background from image]")
        image_data = None
        if rem_bak_file != None:
            image_data = rem_bak_file.getvalue()
            filename = "bkgd_upload_image.png"
            with open(filename, 'wb') as file:
                file.write(image_data)
            image = Image.open("bkgd_upload_image.png")

        if rem_bak_file is not None:
            pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", revision ="refs/pr/9", trust_remote_code=True, device='cpu')
            pipe("bkgd_upload_image.png", out_name=f"finished_remove.png")
            st.write(':green[finished removing background!]')
    try:
        st.image("finished_remove.png")
        os.remove("finished_remove.png")
    except:
        pass
else:
    st.image("./llmonpy/pie.png", caption="please login to continue")