from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import subprocess
import numpy as np 
from datetime import datetime
import cv2
import face_recognition
import time
from datetime import datetime
import csv

app = Flask(__name__)

# Mock data for demonstration
attendance_data = pd.DataFrame(columns=['Name', 'Time', 'Date'])

def run_jupyter_notebook(notebook_path):
    try:
        subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', notebook_path], check=True)
        print("Notebook execution successful.")
    except subprocess.CalledProcessError as e:
        print(f"Notebook execution failed with error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request has the file part
    if 'image' not in request.files:
        # Redirect to the home page if no file is uploaded
        return redirect(url_for('home'))

    file = request.files['image']

    # Check if the file is empty
    if file.filename == '':
        # Redirect to the home page if no file is selected
        return redirect(url_for('home'))

    # Check if 'DataSet' directory exists; if not, create it
    if not os.path.exists('DataSet'):
        os.makedirs('DataSet')

    # Save the file to the 'DataSet' folder
    file.save(os.path.join('DataSet', file.filename))

    # Add your logic to update the DataSet with the new image
    # For example, you can use face recognition to encode and store the face

    # Redirect to the home page after processing
    return redirect(url_for('home'))

@app.route('/view_attendance', methods=['GET'])
def view_attendance():
    # Read the attendance CSV file
    try:
        df = pd.read_csv("Attendance.csv", on_bad_lines='skip')
        attendance_data = df.to_html()
    except FileNotFoundError:
        attendance_data = "<p>No attendance data available.</p>"

    return render_template('attendance.html', data=attendance_data)


@app.route('/redirect_to_attendance')
def redirect_to_attendance():
    return redirect(url_for('view_attendance'))
# New route for starting attendance with webcam
@app.route('/start_attendance', methods=['GET'])
def start_attendance():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('webcam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('webcam', 800, 600)

    # Load known faces and names from the dataset
    path = 'DataSet'
    images = []
    classNames = []
    mylist = os.listdir(path)

    for cl in mylist:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    encodeListKnown = [face_recognition.face_encodings(img)[0] for img in images]

    start_time = time.time()  # Record the start time

    while True:
        # Read a frame from the webcam
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

        for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
            matches = face_recognition.compare_faces(encodeListKnown, encode_face)
            faceDist = face_recognition.face_distance(encodeListKnown, encode_face)
            matchIndex = np.argmin(faceDist)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper().lower()
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                # Mark attendance in CSV file
                mark_attendance(name)

        cv2.imshow('webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check if 10 seconds have passed
        elapsed_time = time.time() - start_time
        if elapsed_time >= 5:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Display success message
    return "Attendance completed successfully."


# Function to mark attendance in CSV file
def mark_attendance(name):
    csv_file_path = 'Attendance.csv'

    # Read existing data from CSV file
    existing_data = pd.read_csv(csv_file_path)

    # Check if the person's entry already exists
    if name.lower() not in existing_data['Name'].str.lower().values:
        with open(csv_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            now = datetime.now()
            time_str = now.strftime('%I:%M:%S:%p')
            date_str = now.strftime('%d-%B-%Y')
            writer.writerow([name, time_str, date_str])

@app.route('/submit_attendance', methods=['POST'])
def submit_attendance():
    # Handle the form submission for submitting attendance
    if request.method == 'POST':
        # Add your attendance submission logic here
        name = request.form.get('student_name')  # Change 'student_name' to your actual form field name

        # Mock submission to the DataFrame
        #global attendance_data
        #attendance_data = attendance_data.append({'Name': name, 'Time': time, 'Date': date}, ignore_index=True)

        # Example usage:
        notebook_path = r'Machine Learning\Untitled.ipynb'
        run_jupyter_notebook(notebook_path)

        # Redirect to the home page after processing
        return redirect(url_for('start_attendance'))
if __name__ == "__main__":
    app.run(debug=True)
