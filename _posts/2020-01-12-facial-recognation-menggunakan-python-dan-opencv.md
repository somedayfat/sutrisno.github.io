---
title: "Facial Recognation Menggunakan Python & OpenCV"
categories:
  - Blog
tags:
  - Face Recognation
  - Python
  - TensorFlow
  - Haarcascade
  - Numpy
  - OpenCV
---
Hi,

Untuk kali ini saya akan menuliskan blog tentang facial recognation. Menggunakan Python dan library-nya

**Hal Yang Perlu Disiapkan**
- Python 3.7 64 Bit
- PyCharm IDE
- Numpy
- OpenCV
- TensorFlow
<h3>Dataset</h3>
- facial_expression_model_sructure.json
- facial_expression_model_weight.h5
- haarcascade_frontalface_default.xml
Bisa didownload [Disini][dataset]

Kemudian taruh dataset tersebut bersamaan dengan program yang akan kita buat.
{% include figure image_path="/assets/images/konten/facial/struktur.png" alt="this is a placeholder image" %}

Bagi yang sudah memempunyai atau sudah mendownload Python bisa langsung mendownload 3 libary tersebut.

Untuk melakukan deteksi wajah sendiri kita akan menggunakan metode haar cascade classifier, metode tersebut bisa kalian pelajari
seara singkat di link [berikut][brkt]. Haar cascade sendiri membutuhkan data training untuk dijadikan sebuah acuan proses classifier dan disini kita akan menggunakan  'haarcascade_frontalface_default.xml', Selain itu kita juga membutuhkan sebuah mode atau algoritma CNN (Convolutional Neural Netwok) untuk mendeteksi atau memprediksi emosi dari wajah yang akan kita klasifikasikan, dan disini kita akan menggunakan 'facial_expression_model_sructure.json'. dan library Keras yang ada pada TensorFlow sendiri akan digunakan untuk membuka model atau algoritma CNN face expression.

**Oke Langsung Saja Kita Mulai**

Buka PyCharm anda dan dan buat file dengan nama latihan.py
kemudian tuliskan kode berikut ini.

1.Inisasi library
```ruby
import cv2
import numpy as np
from keras_preprocessing import image
from tensorflow.keras.models import model_from_json
```
<h3> Numpy </h3> Akan digunakan untuk proses operasi vektor dan matrix dengan mengolah array dan array multidimensi.
<h3>cv2(OpenCV)</h3> Akan kita gunakan untuk mengolah gambar dan video yang nantinya akan digunakan untuk mengolah informasi pada media tersebut.
<h3>Keras & TensorFlow</h3> Library yang akan digunakan dalam pembelajaran Machine Learning ini sendiri.

2.Capture video

Tambahkan kode dibawah ini untuk menjalankan webcam yang ada pada laptop kita.

```ruby
cap = cv2.VideoCapture(0) #Kelas yang akan digunakan untuk mengambil gambar
while (True):

    ret, img = cap.read()
    cv2.imshow('img', img)
#Digunakan untuk menampilkan hasil kamera

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#Digunakan untuk keluar dari kamera dengan menekan tombol 'Q'

cap.release()
cv2.destroyAllWindows()
```
3.Menggunakan Haar Cascade Classifier

Setelah sebelumnya kita baru mencoba untuk menampilkan kamera menggunakan webcam. Selanjutnya
kita akan mendeteksi wajah dengan HCC, Tambahkan kode berikut:

```ruby
import cv2
import numpy as np
from keras_preprocessing import image
from tensorflow.keras.models import model_from_json

face_cascade = cv2.CascadeClassifier(r'model\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
#Membuka file haarcascade_frontalface_default.xml

while(True):
    ret, img = cap.read()
    cv2.imshow('img', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#Mengubah hasi tangkapan kamera menjadi hitam putih kemudian akan dideteksi menggunakan HCC dan selanjutnya hasil pendeteksian akan diberi box berwarna biru
    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break


cap.release()
cv2.destroyAllWindows()
```
Algoritma Haar Cascade sendiri merupakan sebuah algoritma deteksi obyek yang menggunakan pendekatan berbasis
Cascade. Dimana algoritma ini sendiri dilatih menggunakan banyak sekali gambar positif & negatif.

4.Load Fungsi CNN Face Expression menggunakan Library keras
```ruby
import cv2
import numpy as np
from keras_preprocessing import image
from tensorflow.keras.models import model_from_json

face_cascade = cv2.CascadeClassifier(r'model\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

model = model_from_json(open(r'model\facial_expression_model_structure.json').read())
model.load_weights(r'model\facial_expression_model_weights.h5')
#Digunakan untuk me-load fungsi Face Exspression yang telah dibuat menggunakan TensorFlow

emotions = ('Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Biasa') # Kemudian di set exspressi yang akan di klasifikasikan
while(True):
    ret, img = cap.read()
    cv2.imshow('img', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
5.Memasukan hasil deteksi wajah ke CNN Face Exspression

```ruby
mport cv2
import numpy as np
from keras_preprocessing import image
from tensorflow.keras.models import model_from_json

face_cascade = cv2.CascadeClassifier(r'model\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
model = model_from_json(open(r'model\facial_expression_model_structure.json').read())
model.load_weights(r'model\facial_expression_model_weights.h5')

emotions = ('Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Biasa')
while(True):
    ret, img = cap.read()
    cv2.imshow('img', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        detected_face = img[int(y):int(y + h), int(x):int(x + w)] #Wajah yang terdeteksi kemudian akan di crop
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #Kemudian akan di ubah formatnya dari RGB ke Hitam Putih
        detected_face = cv2.resize(detected_face, (48, 48))#Dan akan diubah ukuranya menjadi 48x48 pixel

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        print(img_pixels)
#Kemudian hasil resize diubah menjadi gambar yang mempunyai range [0-255] menjadi range [0-1] (Menyesuaikan range nilai klasifikasi )

        img_pixels /= 255

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotion = emotions[max_index]
#Data yang telah disesuaikan rangenya kemudian akan diprediksi emosinya.
        cv2.putText(img, emotion, (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#Kemudian hasil ekpresi yang telah diprediksi akan ditampilkan dilayar

    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
{% include figure image_path="/assets/images/konten/facial/facial.png" alt="this is a placeholder image" caption="Hasil Facial Recognation HCC" %}

**Selesai**

Untuk Datasetnya sendiri anda bisa mendownloadnya [DISINI][dataset]


[brkt]:http://tinyurl.com/s46vthr
[dataset]:https://drive.google.com/file/d/1kIFON73puyzXiHi72bBJ1U_pnlGKolZR/view?usp=sharing
