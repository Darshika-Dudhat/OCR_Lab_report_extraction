import streamlit as st
import pandas as pd 
import pytesseract
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np 
import os
from datetime import datetime
import re

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLO model
model = YOLO("./models/lab_table_ocr/weights/best.pt")

# App Title
st.title("Lab Report OCR Extraction")

# Upload file
uploaded_file = st.file_uploader("Upload Lab Report Image", type=["jpg", "jpeg","png"])

def clean_patient_name(text):
    # Remove common noise like digits or timestamps before actual name
    return re.sub(r"^[^A-Za-z]*", "", text).strip()

if uploaded_file is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    st.image(image_np, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Running OCR..."):
        results = model(image_np)[0]
        
        # labels_detected = [results.names[int(box[-1])] for box in results.boxes.data]
        # st.write("Detected labels:", labels_detected)
        
        data = {
            "PATIENT_NAME": [],
            "TEST_NAME": [],
            "VALUE": [],
            "UNIT": [],
            "RANGE": []
        }
        
        label_map = {
            "PATIENT_NAME": "PATIENT_NAME",
            "TEST_NAME": "TEST_NAME",
            "VALUE": "VALUE",
            "UNIT": "UNIT",
            "RANGE": "RANGE"
        }
        
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            label = results.names[int(cls)]
            
            if label not in label_map:
                continue
            
            cropped = image_np[int(y1):int(y2), int(x1):int(x2)]
            gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
            
                
            # # Optional preprocessing for better OCR
            # gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            text = pytesseract.image_to_string(gray, config="--psm 6").strip()
            for line in text.splitlines():
                if line.strip():
                    data[label_map[label]].append(line.strip())
                    
        # Align lengths
        min_len = min(len(data["TEST_NAME"]), len(data["VALUE"]), len(data["UNIT"]), len(data["RANGE"]))

        # Save patient name (if exists)
        patient_name = data["PATIENT_NAME"][0] if data["PATIENT_NAME"] else "Not Found"
        
        patient_name = clean_patient_name(patient_name)
        
        # st.write("Extracted Patient Name Data:", data["PATIENT_NAME"])

        if data["PATIENT_NAME"]:
            st.subheader("Patient Name")
            st.text(patient_name)
        
        df = pd.DataFrame({
            "TEST_NAME" : data["TEST_NAME"][:min_len],
            "VALUE": data["VALUE"][:min_len],
            "UNIT": data["UNIT"][:min_len],
            "RANGE": data["RANGE"][:min_len]
        })
        
        # # Remove duplicate rows
        # df.drop_duplicates(inplace=True)
        # df.reset_index(drop=True, inplace=True)
                
        st.success("OCR Completed")
        st.dataframe(df)
        
        # Download buttons
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "lab_report.csv", "text/csv")

        # Save Excel with two sheets: Patient Info + Lab Results
        safe_name = patient_name.replace(" ", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file_name = f"{safe_name}_{timestamp}.xlsx"
        
        excel_path = os.path.join(tempfile.gettempdir(), excel_file_name)
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            pd.DataFrame({"Patient Name": [patient_name]}).to_excel(writer, sheet_name="Patient Info", index=False)
            df.to_excel(writer, sheet_name="Lab Results", index=False)

        with open(excel_path, "rb") as f:
            st.download_button("Download Excel", f.read(), excel_file_name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")