from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pickle
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Bidirectional, LSTM, RepeatVector, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import os
import pandas as pd
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import tensorflow as tf
import time
from DCGAN import DCGAN
import sys
sys.path.append('./tools/')
from utils import save_images, save_source
from data_generator import ImageDataGenerator
from keras import backend as K

# Global variables
global sess, dcgan_model, tfidf_vectorizer, sc, model, model_graph, keras_session

# Configure TensorFlow session to allow GPU memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# ========================
# Initialize Keras Model Session
# ========================
# Create a dedicated session for Keras and set it
keras_session = tf.Session(config=config)
K.set_session(keras_session)

# Text preprocessing components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def cleanText(doc):
    """
    Clean text by removing punctuation, non-alphabetical tokens, stop words,
    and applying stemming and lemmatization.
    """
    table = str.maketrans('', '', punctuation)
    tokens = doc.lower().translate(table).split()
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words and len(token) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Load data for text-to-image model
X = np.load("model/X.npy")
Y = np.load("model/Y.npy")
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
X = tfidf_vectorizer.fit_transform(X).toarray()
sc = StandardScaler()
X = sc.fit_transform(X)
X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
Y = np.reshape(Y, (Y.shape[0], Y.shape[1] * Y.shape[2] * Y.shape[3]))
Y = Y.astype('float32') / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Build the Keras CNN model for text-to-image generation
model = Sequential()
model.add(Conv2D(32, (1, 1), activation='relu', input_shape=(X.shape[1], X.shape[2], X.shape[3])))
model.add(MaxPooling2D((1, 1)))
model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(MaxPooling2D((1, 1)))
model.add(Conv2D(128, (1, 1), activation='relu'))
model.add(MaxPooling2D((1, 1)))
model.add(Flatten())
model.add(RepeatVector(2))
model.add(Bidirectional(LSTM(128, activation='relu')))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(Y.shape[1], activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

if not os.path.exists("model/cnn_weights.hdf5"):
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose=1, save_best_only=True)
    hist = model.fit(X_train, y_train, batch_size=16, epochs=15, validation_data=(X_test, y_test),
                     callbacks=[model_check_point], verbose=1)
    with open('model/cnn_history.pckl', 'wb') as f:
        pickle.dump(hist.history, f)
else:
    model = load_model("model/cnn_weights.hdf5")
with open('model/cnn_history.pckl', 'rb') as f:
    cnn_history = pickle.load(f)
accuracy_value = cnn_history['accuracy'][-1]

# Capture the graph used by the Keras model
model_graph = keras_session.graph

# ========================
# Initialize DCGAN Components
# ========================
# Initialize image generators for DCGAN
generator = ImageDataGenerator(batch_size=32, height=128, width=128, z_dim=256, scale_size=(128,128), shuffle=False, mode='train')
val_generator = ImageDataGenerator(batch_size=32, height=128, width=128, z_dim=256, scale_size=(128,128), shuffle=False, mode='test')

# Initialize TensorFlow session and load the DCGAN model
with tf.Graph().as_default():
    sess = tf.Session(config=config)
    dcgan_model = DCGAN(sess=sess, lr=0.001, keep_prob=1.0, model_num=None, batch_size=32,
                        gan_loss_weight=1.0, fea_loss_weight=10.0, age_loss_weight=1.0, tv_loss_weight=0.0)
    dcgan_model.imgs = tf.placeholder(tf.float32, [32, 128, 128, 3])
    dcgan_model.true_label_features_128 = tf.placeholder(tf.float32, [32, 128, 128, 5])
    dcgan_model.ge_samples = dcgan_model.generate_images(dcgan_model.imgs, dcgan_model.true_label_features_128,
                                                          stable_bn=False, mode='train')
    dcgan_model.get_vars()
    dcgan_model.saver = tf.train.Saver(dcgan_model.save_g_vars)
    sess.run(tf.global_variables_initializer())
    if dcgan_model.load(dcgan_model.saver, 'dcgan', 399999):
        print("DCGAN model successfully loaded")

# ========================
# View Functions
# ========================
def generateDetectFake(filename):
    """
    Generate fake images using the DCGAN model from an input image.
    """
    arr = ['original', 'fake1', 'fake2', 'fake3', 'fake4']
    img_list = []
    source = val_generator.load_imgs(filename, 128)
    train_imgs = generator.load_train_imgs("CelebDataset/train", 128)
    temp = np.reshape(source, (1, 128, 128, 3))
    save_source(temp, [1, 1], "output/" + arr[0] + ".jpg")
    images = np.concatenate((temp, train_imgs), axis=0)
    for j in range(1, generator.n_classes):
        true_label_fea = generator.label_features_128[j]
        feed_dict = {dcgan_model.imgs: images, dcgan_model.true_label_features_128: true_label_fea}
        samples = sess.run(dcgan_model.ge_samples, feed_dict=feed_dict)
        image = np.reshape(samples[0, :, :, :], (1, 128, 128, 3))
        save_images(image, [1, 1], "output/" + arr[j] + ".jpg")
    orig = cv2.imread("output/" + arr[0] + ".jpg")
    orig = cv2.resize(orig, (250, 250))
    for i in range(len(arr)):
        img = cv2.imread("output/" + arr[i] + ".jpg")
        img = cv2.resize(img, (250, 250))
        cv2.putText(img, arr[i], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        img_list.append(img)
    images_concat = cv2.hconcat(img_list)
    return images_concat

def HumanFacesAction(request):
    """
    Process an uploaded human face image and generate fake images using the DCGAN model.
    """
    if request.method == 'POST':
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        filepath = 'ImageApp/static/' + fname
        if os.path.exists(filepath):
            os.remove(filepath)
        with open(filepath, "wb") as file:
            file.write(myfile)
        img = generateDetectFake(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        context = {'data': "DCGAN Generated Fake Images", 'img': img_b64}
        return render(request, 'ViewResult.html', context)

def HumanFaces(request):
    if request.method == 'GET':
        return render(request, 'HumanFaces.html', {})

def PredictPerformance(request):
    if request.method == 'GET':
        return render(request, 'PredictPerformance.html', {})

def TexttoImageAction(request):
    """
    Generate an image based on text input using the trained CNN model.
    """
    if request.method == 'POST':
        text_data = request.POST.get('t1', '').strip()
        if not text_data:
            context = {'data': "No text provided.", 'img': ""}
            return render(request, 'ViewResult.html', context)
        cleaned_text = cleanText(text_data)
        text_vector = tfidf_vectorizer.transform([cleaned_text]).toarray()
        text_vector = sc.transform(text_vector)
        text_vector = np.reshape(text_vector, (text_vector.shape[0], text_vector.shape[1], 1, 1))
        with model_graph.as_default():
            with keras_session.as_default():
                prediction = model.predict(text_vector)
        prediction = prediction[0]
        prediction = np.reshape(prediction, (128, 128, 3))
        prediction = cv2.resize(prediction, (300, 300))
        img = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        context = {'data': "Text = " + text_data, 'img': img_b64}
        return render(request, 'ViewResult.html', context)

def TrainModel(request):
    """
    Display training performance graphs using saved training history.
    """
    if request.method == 'GET':
        with open("model/cnn_history.pckl", 'rb') as f:
            train_history = pickle.load(f)
        acc_value = train_history.get('accuracy', [])
        loss_value = train_history.get('loss', [])
        plt.figure(figsize=(6, 4))
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.plot(acc_value, 'o-', color='green', label='Accuracy')
        plt.plot(loss_value, 'o-', color='blue', label='Loss')
        plt.legend(loc='upper left')
        plt.title('DCGAN Training Accuracy & Loss')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        context = {'data': '<font size="3" color="blue">DCGAN Accuracy: ' + str(accuracy_value) + '</font>',
                   'img': img_b64}
        return render(request, 'ViewResult.html', context)

def AdminLogin(request):
    if request.method == 'GET':
        return render(request, 'AdminLogin.html', {})

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})

def AdminLoginAction(request):
    """
    Process admin login.
    """
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()
        if username == "admin" and password == "admin":
            context = {'data': 'Welcome ' + username}
            return render(request, 'AdminScreen.html', context)
        else:
            context = {'data': 'Invalid login details'}
            return render(request, 'AdminLogin.html', context)

def TexttoImage(request):
    if request.method == 'GET':
        return render(request, 'TexttoImage.html', {})

# End of views.py
