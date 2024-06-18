from flask import Flask, render_template, request, send_file
from PIL import Image
import numpy as np
import cv2  
import glob
import pickle
import pandas as pd
from tensorflow import keras
from tensorflow import image
from PIL import Image
from tensorflow.keras.preprocessing import image
import tensorflow 
import time 
import os 
import paddle
import argparse
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel
from tqdm import tqdm

app = Flask(__name__)

##GENERATION PART
def get_id_emb(id_net, id_img): 
    id_img = cv2.resize(id_img, (112, 112)) 
    id_img = cv2paddle(id_img) 
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std 
    id_emb, id_feature = id_net(id_img) 
    id_emb = l2_norm(id_emb) 
    return id_emb, id_feature 





def video_test(source_img_path, target_video_path, output_path, image_size, merge_result, use_gpu):
    paddle.set_device("gpu" if use_gpu else 'cpu')
    faceswap_model = FaceSwap(use_gpu) 

    id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3]) 
    id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams')) 
    id_net.eval() 

    weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams') 

    landmarkModel = LandmarkModel(name='landmarks') 
    landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640))
    id_img = cv2.imread(source_img_path) 

    landmark = landmarkModel.get(id_img) 
    if landmark is None:
        print('**** No Face Detect Error ****')
        exit()

    aligned_id_img, _ = align_img(id_img, landmark)
    id_emb, id_feature = get_id_emb(id_net, aligned_id_img) 

    faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
    faceswap_model.eval()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture()
    cap.open(target_video_path)
    videoWriter = cv2.VideoWriter(os.path.join(output_path, os.path.basename(target_video_path)), fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
   
    all_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in tqdm(range(int(all_f))): 
        ret, frame = cap.read()
        landmark = landmarkModel.get(frame) 
        if landmark is not None:
            att_img, back_matrix = align_img(frame, landmark) 
            att_img = cv2paddle(att_img) 
            res, mask = faceswap_model(att_img) 
            res = paddle2cv(res) 
            mask = np.transpose(mask[0].numpy(), (1, 2, 0)) 
            res = dealign(res, frame, back_matrix, mask) 
            frame = res 
        else:
            print('**** No Face Detect Error ****')
        videoWriter.write(frame)
    cap.release()
    videoWriter.release()

@app.route("/")
def home():
    return render_template("temp.html")

@app.route('/generate.html')
def generate():
    return render_template("generate.html") 

@app.route('/Generator1', methods=['POST'])
def Generator1(): 
    global video1 
    video1 = request.files['video1']
    if video1.filename[-3:] in ["mp4","jpg","png"]:
        # Create the directory if it doesn't exist
        os.makedirs('generate/input/', exist_ok=True)
        video1.save('generate/input/' + video1.filename)
    return render_template("generate.html")

@app.route('/Generator2', methods=['POST'])
def Generator2():  
    global video1 
    video2 = request.files['video2']
    if video2.filename[-3:] in ["mp4", "jpg", "png"]:
        video2.save('generate/input/' + video2.filename)
    source_img_path = "E:/GraduationProject/generate/input/" + video2.filename
    target_video_path = "E:/GraduationProject/generate/input/" + video1.filename
    output_path = "E:/GraduationProject/result"
    image_size = 256
    merge_result = True
    use_gpu = False
    print(video1.filename)
    video_test(source_img_path, target_video_path, output_path, image_size, merge_result, use_gpu)
    
    return render_template("generate.html")

@app.route('/download-video')
def download_video():
    video_path = f'E:/GraduationProject/result/{os.listdir("result/")[0]}'
    return send_file(video_path, as_attachment=True, mimetype='video/mp4')

#DETECTION PART
ALLOWED_EXTENSIONS = ['mp4']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 

@app.route('/upload', methods=['POST']) 
def upload():
    video = request.files['video']

    for i in os.listdir(f"E:/GraduationProject/faceimg/"):
        os.remove(f"E:/GraduationProject/faceimg/{i}")
        
    if video.filename[-3:]=="mp4": 
        video.save('static/videos/' + video.filename) 
        img_number=0
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") 
        path='static/videos/' + video.filename  
        all_videos=glob.glob(path)
        li=[]
        counter=0
        for video in all_videos:
            cap = cv2.VideoCapture(video)  
            counter=counter+1
            print(counter)
            _, img = cap.read()     
            _, img = cap.read()
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=2)  
            wmax=0 
            hmax=0
            ImageFolder ='faceimg/'
            for (x, y, w, h) in faces: 
                if w>wmax and h>hmax:
                    wmax=w
                    hmax=h

            for (x, y, w, h) in faces: 
                img_number=img_number+1
                if w == wmax and h==hmax : 
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1) 
                    crop_img = img[y:y+h, x:x+w]
                    li.append(crop_img)
                    cv2.imwrite("faceimg/"+"image"+".jpg",crop_img) 

            k = cv2.waitKey(1000) & 0xff  
            if k==27:  
                break  
                    
        cap.release()
        print("preload")

    else:
        video.save(f"E:/GraduationProject/faceimg/{video.filename}")  
    
    pickled_model = tensorflow.keras.models.load_model("E:/GraduationProject/cnn2.hdf5", compile=False) 
    print("precompile")

    pickled_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy']) 
    print("postcompile") 

    img = image.load_img(f'faceimg/{os.listdir("faceimg/")[0]}', target_size=(224, 224)) 
    x = image.img_to_array(img) 
    x = np.expand_dims(x, axis=0)
    x=x/255 
    predict_x=pickled_model.predict(x)  
    if predict_x[0][0] > 0.5: 
        print(predict_x[0])
        prediction="fake"
    else:
        print(predict_x[0])
        prediction="real"    

    return render_template('temp.html', prediction=prediction)   

if __name__ == "__main__": 
    app.debug = True
    app.run()
