import cv2
import numpy as np
import streamlit as st
import urllib.request

# Fungsi untuk mendeteksi wajah
def detectFace(net, frame, confidence_threshold=0.7):
    frameOpencvDNN = frame.copy()
    frameHeight = frameOpencvDNN.shape[0]
    frameWidth = frameOpencvDNN.shape[1]
    # Membuat blob dari frame gambar
    blob = cv2.dnn.blobFromImage(frameOpencvDNN, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDNN, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDNN, faceBoxes

# URL repositori GitHub untuk model-model
base_url = 'https://github.com/risya1411/age-detection/models/'

# Nama file-model yang akan digunakan
faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'
ageProto = 'age_deploy.prototxt'
ageModel = 'age_net.caffemodel'
genderProto = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'

# Unduh file-model dari GitHub
urllib.request.urlretrieve(base_url + faceProto, faceProto)
urllib.request.urlretrieve(base_url + faceModel, faceModel)
urllib.request.urlretrieve(base_url + ageProto, ageProto)
urllib.request.urlretrieve(base_url + ageModel, ageModel)
urllib.request.urlretrieve(base_url + genderProto, genderProto)
urllib.request.urlretrieve(base_url + genderModel, genderModel)

# List untuk gender dan usia
genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Membaca model menggunakan OpenCV DNN
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Streamlit app
st.title("Age and Gender Detection")
st.write("Upload an image to detect age and gender")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    resultImg, faceBoxes = detectFace(faceNet, image)
    
    if not faceBoxes:
        st.write("No face detected")
    else:
        for faceBox in faceBoxes:
            # Membuat ROI (Region of Interest) untuk wajah
            face = image[max(0, faceBox[1] - 20):min(faceBox[3] + 20, image.shape[0] - 1),
                         max(0, faceBox[0] - 20):min(faceBox[2] + 20, image.shape[1] - 1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
            
            # Prediksi gender
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            # Prediksi usia
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            # Menambahkan label ke gambar hasil
            label = f'{gender}, {age}'
            cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Menampilkan gambar hasil di Streamlit
        st.image(cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB), caption='Processed Image', use_column_width=True)