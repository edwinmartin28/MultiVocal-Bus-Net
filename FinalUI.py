#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import os
import cv2
#from google.colab.patches import cv2_imshow
import easyocr
from ultralytics import YOLO
from gtts import gTTS
from IPython.display import Audio, display
import pandas as pd
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import streamlit as st
import base64
import time
number_plate_detected = False
# Dictionary to store the last announcement time for each bus
last_announcement_time = []
flag=0

def delete_value(value, last_announcement_time):
    if value in last_announcement_time:
        last_announcement_time.remove(value)
        print(f"Deleted {value} from the list.")

# Function to announce text with a delay
def announce_with_delay(text, text_to_check, language='en'):
    global flag
    if text_to_check in last_announcement_time:
        return None
    else:
        last_announcement_time.append(text_to_check)
        
    
    # Generate audio files for all languages
    text_to_speak_en = text
    text_to_speak_ml = text
    text_to_speak_hi = text
    
    output_audio_en = text_to_speech(text_to_speak_en, language='en')
    output_audio_ml = text_to_speech(text_to_speak_ml, language='ml')
    output_audio_hi = text_to_speech(text_to_speak_hi, language='hi')
    flag=1
    # Display audio files
    display(output_audio_en)
    time.sleep(4)
    display(output_audio_ml)
    time.sleep(3)
    display(output_audio_hi)
    
    # Start a timer to delete the value from the list after 300 seconds
    threading.Timer(180, delete_value, args=(text_to_check, last_announcement_time)).start()

    
# Function to check text in CSV file and retrieve corresponding values
def check_text_and_retrieve_values(csv_file, column_to_search, text_to_check):
    global flag
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Check if the text is present in the specified column
    if df[column_to_search].str.contains(fr"\b{text_to_check}\b").any():
        # If text is found, get the values from the same row
        row_with_text = df[df[column_to_search].str.contains(fr"\b{text_to_check}\b")]
        column4_value = row_with_text['DECISION'].iloc[0]
        if column4_value == 1:
            value = row_with_text['STARTING POINT'].iloc[0]
            row_index = row_with_text.index[0]
            df.at[row_index, 'DECISION'] = 0
        elif column4_value == 0:
            value = row_with_text['DESTINATION'].iloc[0]
            row_index = row_with_text.index[0]
            df.at[row_index, 'DECISION'] = 1
        print(f"The text '{text_to_check}' is present in column '{column_to_search}'.")
        print(f"Value: {value}")
        flag=0
        # Generate audio files for English, Malayalam, and Hindi
        text_to_speak_en = value
        text_to_speak_ml = value
        text_to_speak_hi = value
        # Announce the text with a delay
        announce_with_delay(value,text_to_check)
        if flag==1:
            # Write the updated DataFrame back to the CSV file
            df.to_csv(csv_file, index=False)
            print("CSV file updated successfully.")            
    else:
        print(f"The text '{text_to_check}' is not present in column '{column_to_search}'.")
        return None

# Function to convert text to speech
def text_to_speech(text, language='en', output_file='output.mp3'):
    if language == 'en':
        # English text
        english_text = f"Bus going to {text} has reached the station"
        tts = gTTS(text=english_text, lang='en', slow=False)
    elif language == 'ml':
        # Malayalam text
        malayalam_text = f"{text} ലേക്ക് പോകുന്ന ബസ് സ്റ്റേഷനിൽ എത്തി"
        tts = gTTS(text=malayalam_text, lang='ml', slow=False)
    elif language == 'hi':
        # Hindi text
        hindi_text = f"{text} जाने वाली बस स्टेशन पहुंच गई है"
        tts = gTTS(text=hindi_text, lang='hi', slow=False)
    else:
        raise ValueError("Language not supported")
  # Save audio file
    output_file = "output.mp3"
    tts.save(output_file)

    # Generate HTML to play audio
    audio_html = f'<audio autoplay="autoplay" controls="controls"><source src="data:audio/mp3;base64,{base64.b64encode(open(output_file, "rb").read()).decode()}" type="audio/mp3" /></audio>'
    
    # Display the HTML
    st.write(audio_html, unsafe_allow_html=True)
    time.sleep(3)


# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'Z': '2',
                    'U': '0',
                    'L': '1'}

# Function to process each frame
def process_image(frame, model, reader):
    global number_plate_detected

    # Perform object detection only if a number plate has not been detected
    if not number_plate_detected:
        # Perform object detection
        results = model(frame)

        # Initialize ocr_result
        ocr_result = None

        # Iterate over each result
        for result in results:
            # Check if any detections were made
            if result.boxes is not None:
                # Iterate over each detected object
                for label, bbox, confidence in zip(result.names, result.boxes.xyxy, result.boxes.conf):
                    if label is not None and bbox is not None and confidence is not None:
                        x1, y1, x2, y2 = map(int, bbox)
                        # Extend the bounding box slightly to ensure the full text is included
                        padding = 10
                        x1 = max(x1 - padding, 0)
                        y1 = max(y1 - padding, 0)
                        x2 = min(x2 + padding, frame.shape[1])
                        y2 = min(y2 + padding, frame.shape[0])
                        # Crop the region defined by the adjusted bounding box
                        cropped_img = frame[y1:y2, x1:x2]
                        # Use EasyOCR to recognize text
                        result = reader.readtext(cropped_img)
                        if result:
                            # Extract the text from all the detected areas
                            texts = [res[1] for res in result]
                            # Combine the extracted texts into a single string
                            text = ' '.join(texts)
                            # Remove special characters and spaces
                            text = ''.join(e for e in text if e.isalnum()).upper()
                            # Check if text length is 9
                            if len(text) == 9:
                                # Correct predictions according to dict_char_to_int
                                corrected_text = ''.join(dict_char_to_int[char] if char in dict_char_to_int else char for char in text[5:])
                                text1 = text[:5] + corrected_text
                                if text1[5:].isdigit():
                                    # Force the first five characters to be 'KL15A'
                                    text = 'KL15A' + text1[5:]
                                    ocr_result = text
                            elif len(text) >= 6:
                                # Correct predictions according to dict_char_to_int
                                corrected_text = ''.join(dict_char_to_int[char] if char in dict_char_to_int else char for char in text[5:])
                                text1 = text[:5] + corrected_text
                                # Check if the rest of the characters are alphanumeric
                                if text1[4].isalpha():
                                    # Force the first five characters to be 'KL15A'
                                    text = 'KL15A' + text1[5:]
                                    ocr_result = text
                                elif text1[4:].isdigit():
                                    # Force the first four characters to be 'KL15'
                                    text = 'KL15' + text1[4:]
                                    ocr_result = text
                        else:
                            print("No text detected.")
                        print("Detected Text:", ocr_result)

                        # Calling the function for retrieving destination and for announcing
                        if check_text_and_retrieve_values("DATA - Sheet.csv", "NUMBER PLATE",
                                                          ocr_result):
                            number_plate_detected = True  # Set the flag to True if number plate is detected
                            return ocr_result  # Terminate the program if number plate is detected

                        # Draw adjusted bounding box and label on the original image
                        label = f"{label} ({confidence:.1f})"  # Reduced to 1 decimal place
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Decreased thickness
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)  # Decreased font size and thickness
                        # Increased font size and thickness
                    else:
                        print("One of the attributes (label, bbox, confidence) is None.")
            else:
                print("No detections made.")

        return None  # Return None if number plate is not detected



# Initialize the YOLO model with the weights file
model = YOLO('best.pt')

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to capture video from webcam
def webcam_capture():
    global number_plate_detected

    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open webcam")
        return

    # Initialize the timer
    start_time = time.time()

    # Placeholder for displaying the camera feed
    camera_placeholder = st.empty()

    # Loop to continuously capture frames from the webcam
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame is captured successfully
        if not ret:
            st.error("Error: Could not read frame")
            break

        # Process the frame
        number_plate = process_image(frame, model, reader)
        if number_plate:
            # Play announcement
            text_to_speech(number_plate)
            number_plate_detected = True  # Set the flag to True if number plate is detected
            start_time = time.time()  # Reset the timer

        # Display the processed frame
        camera_placeholder.image(frame, channels="BGR")

        # Check if the timer has exceeded 10 seconds
        if time.time() - start_time >= 5:
            number_plate_detected = False  # Reset the flag
            start_time = time.time()  # Reset the timer

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
# Set page configuration to display the background image
st.set_page_config(page_title="Multilingual Automated Bus Route Announcement System", page_icon=":bus:", layout="wide", initial_sidebar_state="auto")

def main():
    st.title("Automated Multilingual Bus Route Announcement System : For KSRTC Buses")
    if st.button("Start Live Feed"):
        webcam_capture()

# Run the Streamlit app
if __name__ == "__main__":
    main()

