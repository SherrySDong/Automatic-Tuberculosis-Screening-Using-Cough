
import numpy as np
import lightgbm as lgb
FILE=open('train_gs','r')
train_data=[]
train_gs=[]
for line in FILE:
    line=line.strip()
    table=line.split('\t')
    cross=np.load('/home/shdong/cough/data/code/solicited_mfcc_norm/' + table[0]+ '.npy')
    data=np.concatenate((cross,np.array([float(table[2])])))#.reshape(1,1)))
    cross=np.load('/home/shdong/cough/data/code/solicited_magnitude_norm/' + table[0]+ '.npy')
    data=np.concatenate((data,cross))#.reshape(1,1)))
    cross=np.load('/home/shdong/cough/data/code/solicited_chroma/' + table[0]+ '.npy')
    data=np.concatenate((data,cross))#.reshape(1,1)))
    cross=np.load('/home/shdong/cough/data/code/solicited_chroma_norm_delta/' + table[0]+ '.npy')
    data=np.concatenate((data,cross))#.reshape(1,1)))
    cross=np.load('/home/shdong/cough/data/code/solicited_delta/' + table[0]+ '.npy')
    data=np.concatenate((data,cross))#.reshape(1,1)))
    cross=np.load('/home/shdong/cough/data/code/solicited_delta2/' + table[0]+ '.npy')
    data=np.concatenate((data,cross))#.reshape(1,1)))
    train_data.append(data)
    train_gs.append(float(table[1]))
FILE.close()

lgb_train = lgb.Dataset(np.asarray(train_data), np.asarray(train_gs))

params = {
    'boosting_type': 'gbdt',
    'num_boost_round':1000
    'learning_rate': 0.04,
    'objective': 'regression',
    'verbose': 0,
    'n_estimators': 500,
    'reg_alpha': 2.5,
}

gbm = lgb.train(params,
    lgb_train
)


import pickle
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools
pickle.dump(gbm, open('model.pkl', 'wb'))
initial_type = [('float_input', FloatTensorType([None, np.asarray(train_data).shape[1]]))]
onnx_model = onnxmltools.convert_lightgbm(gbm, initial_types=initial_type)

onnxmltools.utils.save_model(onnx_model, 'model.onnx')

