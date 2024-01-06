import tkinter as tk
from tkinter import filedialog,messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from flask import Flask, jsonify

with tf.device('/cpu:0'):
    model = load_model('./model.h5')

def process_image():
    image_path = filedialog.askopenfilename()
    if(image_path != None):
        img = cv2.imread(image_path, 1)
        img_resized = cv2.resize(img, (150, 150))
        img_resized = img_resized.astype(np.float32) / 255.0 
        prediction = model.predict(np.array([img_resized]))[0][0:4]
        messagebox.showinfo("Process", f"Image output is {prediction}")

window = tk.Tk()
buttoon_select = tk.Button(window,text="Select Image...",command=process_image)
buttoon_select.pack(pady=10)
window.mainloop() 