import os
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import load_model

#TODO:Datset formation
# data_lst=[]
# age_lst=[]
# data_pth=r'C:\Users\HP\OneDrive\Desktop\PranCode\Datsets\face_age'
# for data_folder in os.listdir(data_pth):
#     data_folder_pth=data_pth+'\\'+data_folder
#     for folder in os.listdir(data_folder_pth):
#         folder_pth=data_folder_pth+'\\'+folder
#         for pik in os.listdir(folder_pth):
#             pik_pth=folder_pth+'\\'+pik
#             age_lst.append(int(data_folder))
#             img=Image.open(pik_pth).convert('RGB')
#             img=img.resize((100,100))
#             img=np.array(img)
#             img=img/255
#             data_lst.append(img)
# data_lst=np.stack(data_lst)
# np.save('Datsets\Face_age.npy',data_lst)
# with open(r'Pickles\Face_age.pkl','wb') as f:
#     pickle.dump(age_lst,f)

#TODO:Dataset Extraction 
# data=np.load('Datsets\Face_age.npy')
# # print(data.shape)
# # (9770, 100, 100, 3)
# with open('Pickles\Face_age.pkl','rb') as f:
#     labels=pickle.load(f)
# # labels=pd.DataFrame(labels)
# # print(labels.value_counts())
# labels=np.array(labels)
# # print(labels.shape)
# # # # (9770,)
# labels=labels.reshape(-1,1)
# # print(np.max(labels))
# # 18  
# # print(labels.shape)
# # # (9770, 1)

# # #TODO:Permute
# index=np.random.permutation(9769)
# data=data[index]
# labels=labels[index]
# # print(labels[0])
# # plt.imshow(data[0])
# # plt.show()

# #TODO:Label formulation
# #!  AGE=(label*5,(label+1)*5]

# # #TODO:Data Divison
# train_x,test_x,train_y,test_y=train_test_split(data,labels,random_state=42,test_size=0.2)
# # print(train_x.shape)
# # print(train_y.shape)
# # print(test_x.shape)
# # print(test_y.shape)
# # # (7815, 150, 150, 3)
# # # (7815, 1)
# # # (1954, 150, 150, 3)
# # # (1954, 1)

# # #TODO:Model formation
# model=keras.models.Sequential()

# model.add(keras.layers.Conv2D(filters=64,activation='relu',input_shape=[100,100,3],kernel_size=3,padding='same'))
# model.add(keras.layers.Conv2D(filters=64,activation='relu',kernel_size=3,padding='same'))
# model.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

# model.add(keras.layers.Conv2D(filters=128,activation='relu',kernel_size=3,padding='same'))
# model.add(keras.layers.Conv2D(filters=128,activation='relu',kernel_size=3,padding='same'))
# model.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

# model.add(keras.layers.Conv2D(filters=256,activation='relu',kernel_size=3,padding='same'))
# model.add(keras.layers.Conv2D(filters=256,activation='relu',kernel_size=3,padding='same'))
# model.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

# model.add(keras.layers.Dropout(0.4))
# model.add(keras.layers.Flatten())

# model.add(keras.layers.Dense(units=128,activation='relu'))
# model.add(keras.layers.Dense(units=19,activation='softmax'))

# # #TODO:Model Compilation
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

# # #TODO:Model Fitting
# model.fit(train_x,train_y,epochs=20,validation_data=(test_x,test_y),batch_size=150)
# model.save('Models\Age2.h5')

model=load_model('Models\Age2.h5')
img=Image.open('Piks\suar_sir.jpg').convert('RGB')
img=img.resize((100,100))
img=np.expand_dims(img,axis=0)
img=np.array(img)
img=img/255

pred=model.predict(img)
pred=np.argmax(pred,axis=1)

print(pred)


    




        
