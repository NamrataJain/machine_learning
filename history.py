# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Wed Mar 11 10:31:58 2020)---
runfile('C:/Users/Namrata/.spyder-py3/temp.py', wdir='C:/Users/Namrata/.spyder-py3')

## ---(Wed Apr 15 15:10:01 2020)---
runfile('C:/Users/Namrata/.spyder-py3/Catvsdog.py', wdir='C:/Users/Namrata/.spyder-py3')

## ---(Sun Apr 19 12:20:59 2020)---
train_features_vgg = get_bottleneck_features(vgg_model, 8)
validation_features_vgg = get_bottleneck_features(vgg_model, validation_imgs_scaled)
print('Train Bottleneck Features:', train_features_vgg.shape, 
'\tValidation Bottleneck Features:', validation_features_vgg.shape) 
input_shape = vgg_model.output_shape[1]
model = Sequential()
model.add(InputLayer(input_shape=(input_shape,)))
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-4),
metrics=['accuracy'])
model.summary()   
runfile('C:/Users/Namrata/.spyder-py3/test.py', wdir='C:/Users/Namrata/.spyder-py3')
runfile('C:/Users/Namrata/.spyder-py3/Happyhouse.py', wdir='C:/Users/Namrata/.spyder-py3')
pon install kt_utils
pip install kt_utils
runfile('C:/Users/Namrata/.spyder-py3/Happyhouse.py', wdir='C:/Users/Namrata/.spyder-py3')
pip install kt_utils 1.0.4
runfile('C:/Users/Namrata/.spyder-py3/Happyhouse.py', wdir='C:/Users/Namrata/.spyder-py3')
runfile('C:/Users/Namrata/.spyder-py3/DeepNeuralNetwork.py', wdir='C:/Users/Namrata/.spyder-py3')
runfile('C:/Users/Namrata/.spyder-py3/Happyhouse.py', wdir='C:/Users/Namrata/.spyder-py3')
runfile('C:/Users/Namrata/.spyder-py3/DeepNeuralNetwork.py', wdir='C:/Users/Namrata/.spyder-py3')
pip install dnn

## ---(Tue Apr 21 12:53:59 2020)---
runfile('C:/Users/Namrata/.spyder-py3/DeepNeuralNetwork.py', wdir='C:/Users/Namrata/.spyder-py3')
runfile('C:/Users/Namrata/.spyder-py3/TF_HandsOn.py', wdir='C:/Users/Namrata/.spyder-py3')
pip3 install resnet
pip install resnet50
pip install resnet

## ---(Tue Apr 21 13:49:44 2020)---
runfile('C:/Users/Namrata/.spyder-py3/TF_HandsOn.py', wdir='C:/Users/Namrata/.spyder-py3')
pip install tf.keras.applications.ResNet50
runfile('C:/Users/Namrata/.spyder-py3/TF_HandsOn.py', wdir='C:/Users/Namrata/.spyder-py3')
runfile('C:/Users/Namrata/Anaconda3/lib/site-packages/tensorflow_core/python/keras/engine/sequential.py', wdir='C:/Users/Namrata/Anaconda3/lib/site-packages/tensorflow_core/python/keras/engine')
runfile('C:/Users/Namrata/.spyder-py3/TF_HandsOn.py', wdir='C:/Users/Namrata/.spyder-py3')
python

## ---(Tue Apr 21 14:28:02 2020)---
runfile('C:/Users/Namrata/.spyder-py3/TF_HandsOn.py', wdir='C:/Users/Namrata/.spyder-py3')
runfile('C:/Users/Namrata/.spyder-py3/DigitalStore.py', wdir='C:/Users/Namrata/.spyder-py3')

## ---(Wed Apr 22 10:35:46 2020)---
runfile('C:/Users/Namrata/.spyder-py3/BeginDeepNN.py', wdir='C:/Users/Namrata/.spyder-py3')
runfile('C:/Users/Namrata/.spyder-py3/Zillow_Home_Value_Prediction.py', wdir='C:/Users/Namrata/.spyder-py3')
runfile('C:/Users/Namrata/.spyder-py3/CNN_recognize images.py', wdir='C:/Users/Namrata/.spyder-py3')
runfile('C:/Users/Namrata/.spyder-py3/diabetic_data.py', wdir='C:/Users/Namrata/.spyder-py3')
runfile('C:/Users/Namrata/.spyder-py3/Flower_recognition.py', wdir='C:/Users/Namrata/.spyder-py3')

## ---(Mon Apr 27 09:01:42 2020)---
runfile('C:/Users/Namrata/.spyder-py3/Flower_recognition.py', wdir='C:/Users/Namrata/.spyder-py3')

## ---(Mon May  4 16:16:25 2020)---
runfile('C:/Users/Namrata/.spyder-py3/Employee Retention.py', wdir='C:/Users/Namrata/.spyder-py3')
runfile('C:/Users/Namrata/.spyder-py3/TF_mulivariate.py', wdir='C:/Users/Namrata/.spyder-py3')
runfile('C:/Users/Namrata/.spyder-py3/jetRails_prediction.py', wdir='C:/Users/Namrata/.spyder-py3')
runfile('C:/Users/Namrata/.spyder-py3/Imagenet_TL.py', wdir='C:/Users/Namrata/.spyder-py3')