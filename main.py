import os
import h5py
from keras.layers import Input, Concatenate, Add, Lambda
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as KB
from complexnn import ComplexConv2D

data_dir = 'data/'
input_dict = h5py.File(os.path.join(data_dir, 'input.mat'))
echo1 = input_dict['input'][:, :, :, 0:4:2]
echo1 = echo1.transpose(0, 3, 1, 2)
echo2 = input_dict['input'][:, :, :, :]
echo2 = echo2.transpose(0, 3, 1, 2)

out_dict = h5py.File(os.path.join(data_dir, 'output.mat'))
imaging = out_dict['output'][:, :, :, :]
imaging = imaging.transpose(0, 3, 1, 2)

RDB_count1 = 3
count = 6
growth_rate = 32
RDB_count2 = 3

def RDBlocks(input_rdb):
    li1 = [Lambda(lambda x: x[:, :KB.int_shape(x)[1] // 2, :, :], output_shape=(KB.int_shape(input_rdb)[1]//2, KB.int_shape(input_rdb)[2], KB.int_shape(input_rdb)[3]))(input_rdb)]
    li2 = [Lambda(lambda x: x[:, KB.int_shape(x)[1] // 2:, :, :], output_shape=(KB.int_shape(input_rdb)[1]//2, KB.int_shape(input_rdb)[2], KB.int_shape(input_rdb)[3]))(input_rdb)]
    pas = ComplexConv2D(filters=growth_rate, kernel_size=3, strides=(1, 1), padding='same', activation='relu', data_format='channels_first')(input_rdb)
    pas1 = Lambda(lambda x: x[:, :KB.int_shape(x)[1] // 2, :, :], output_shape=(KB.int_shape(pas)[1]//2, KB.int_shape(pas)[2], KB.int_shape(pas)[3]))(pas)
    pas2 = Lambda(lambda x: x[:, KB.int_shape(x)[1] // 2:, :, :], output_shape=(KB.int_shape(pas)[1]//2, KB.int_shape(pas)[2], KB.int_shape(pas)[3]))(pas)
    for i in range(2, count + 1):
        li1.append(pas1)
        li2.append(pas2)
        out = li1 + li2
        out = Concatenate(axis=1)(out)
        pas = ComplexConv2D(filters=growth_rate, kernel_size=3, strides=(1, 1), padding='same', activation='relu', data_format='channels_first')(out)
        pas1 = Lambda(lambda x: x[:, :KB.int_shape(x)[1] // 2, :, :], output_shape=(KB.int_shape(pas)[1] // 2, KB.int_shape(pas)[2], KB.int_shape(pas)[3]))(pas)
        pas2 = Lambda(lambda x: x[:, KB.int_shape(x)[1] // 2:, :, :], output_shape=(KB.int_shape(pas)[1] // 2, KB.int_shape(pas)[2], KB.int_shape(pas)[3]))(pas)
    li1.append(pas1)
    li2.append(pas2)
    out = li1 + li2
    out = Concatenate(axis=1)(out)
    feat = ComplexConv2D(filters=32, kernel_size=1, strides=(1, 1), padding='same', activation='linear', data_format='channels_first')(out)
    feat = Add()([feat, input_rdb])
    return feat

def RDN(f__1, RDB_count):
    f__2 = ComplexConv2D(32, kernel_size=3, strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(f__1)
    RDB = RDBlocks(f__2)
    RDB1 = Lambda(lambda x: x[:, :KB.int_shape(x)[1] // 2, :, :], output_shape=(KB.int_shape(RDB)[1]//2, KB.int_shape(RDB)[2], KB.int_shape(RDB)[3]))(RDB)
    RDB2 = Lambda(lambda x: x[:, KB.int_shape(x)[1] // 2:, :, :], output_shape=(KB.int_shape(RDB)[1]//2, KB.int_shape(RDB)[2], KB.int_shape(RDB)[3]))(RDB)
    RDBlocks_list1 = [RDB1, ]
    RDBlocks_list2 = [RDB2, ]
    for i in range(2, RDB_count + 1):
        RDB = RDBlocks(RDB)
        RDB1 = Lambda(lambda x: x[:, :KB.int_shape(x)[1] // 2, :, :], output_shape=(KB.int_shape(RDB)[1]//2, KB.int_shape(RDB)[2], KB.int_shape(RDB)[3]))(RDB)
        RDB2 = Lambda(lambda x: x[:, KB.int_shape(x)[1] // 2:, :, :], output_shape=(KB.int_shape(RDB)[1]//2, KB.int_shape(RDB)[2], KB.int_shape(RDB)[3]))(RDB)
        RDBlocks_list1.append(RDB1)
        RDBlocks_list2.append(RDB2)
    out = RDBlocks_list1 + RDBlocks_list2
    out = Concatenate(axis=1)(out)
    out = ComplexConv2D(filters=32, kernel_size=1, strides=(1, 1), padding='same', data_format='channels_first')(out)
    out = ComplexConv2D(filters=32, kernel_size=3, strides=(1, 1), activation='linear', padding='same', data_format='channels_first')(out)
    output = Add()([out, f__1])
    return output

def RDN_temporal(f__1, RDN1, RDB_count):
    f__2 = ComplexConv2D(32, kernel_size=3, strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(f__1)
    RDB = RDBlocks(f__2)
    RDB1 = Lambda(lambda x: x[:, :KB.int_shape(x)[1] // 2, :, :], output_shape=(KB.int_shape(RDB)[1]//2, KB.int_shape(RDB)[2], KB.int_shape(RDB)[3]))(RDB)
    RDB2 = Lambda(lambda x: x[:, KB.int_shape(x)[1] // 2:, :, :], output_shape=(KB.int_shape(RDB)[1]//2, KB.int_shape(RDB)[2], KB.int_shape(RDB)[3]))(RDB)
    RDBlocks_list1 = [RDB1, ]
    RDBlocks_list2 = [RDB2, ]
    for i in range(2, RDB_count + 1):
        RDB = RDBlocks(RDB)
        RDB1 = Lambda(lambda x: x[:, :KB.int_shape(x)[1] // 2, :, :], output_shape=(KB.int_shape(RDB)[1]//2, KB.int_shape(RDB)[2], KB.int_shape(RDB)[3]))(RDB)
        RDB2 = Lambda(lambda x: x[:, KB.int_shape(x)[1] // 2:, :, :], output_shape=(KB.int_shape(RDB)[1]//2, KB.int_shape(RDB)[2], KB.int_shape(RDB)[3]))(RDB)
        RDBlocks_list1.append(RDB1)
        RDBlocks_list2.append(RDB2)
    out = RDBlocks_list1 + RDBlocks_list2
    out = Concatenate(axis=1)(out)
    out = ComplexConv2D(filters=32, kernel_size=1, strides=(1, 1), padding='same', data_format='channels_first')(out)
    out = ComplexConv2D(filters=32, kernel_size=3, strides=(1, 1), activation='linear', padding='same', data_format='channels_first')(out)
    output = Add()([out, RDN1])
    return output

init1 = Input(shape=(2, 50, 50), name='input1')
init2 = Input(shape=(4, 50, 50), name='input2')
f1 = ComplexConv2D(32, kernel_size=3, strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(init1)
f2 = ComplexConv2D(32, kernel_size=3, strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(init2)
RDN1 = RDN(f1, RDB_count1)
RDN2 = RDN(f2, RDB_count1)
RDN1_real = [Lambda(lambda x: x[:, :KB.int_shape(x)[1] // 2, :, :], output_shape=(KB.int_shape(RDN1)[1]//2, KB.int_shape(RDN1)[2], KB.int_shape(RDN1)[3]))(RDN1)]
RDN1_imag = [Lambda(lambda x: x[:, KB.int_shape(x)[1] // 2:, :, :], output_shape=(KB.int_shape(RDN1)[1]//2, KB.int_shape(RDN1)[2], KB.int_shape(RDN1)[3]))(RDN1)]
RDN2_real = Lambda(lambda x: x[:, :KB.int_shape(x)[1] // 2, :, :], output_shape=(KB.int_shape(RDN2)[1]//2, KB.int_shape(RDN2)[2], KB.int_shape(RDN2)[3]))(RDN2)
RDN2_imag = Lambda(lambda x: x[:, KB.int_shape(x)[1] // 2:, :, :], output_shape=(KB.int_shape(RDN2)[1]//2, KB.int_shape(RDN2)[2], KB.int_shape(RDN2)[3]))(RDN2)
RDN1_real.append(RDN2_real)
RDN1_imag.append(RDN2_imag)
out = RDN1_real + RDN1_imag
out = Concatenate(axis=1)(out)
out = RDN_temporal(out, RDN1, RDB_count2)
output = ComplexConv2D(filters=32, kernel_size=3, strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(out)
x = ComplexConv2D(filters=1, kernel_size=3, strides=(1, 1), activation='abs', padding='same', name='imaging', data_format='channels_first')(output)

batch_size = 16
print('Training ------------')
model = Model(inputs=[init1, init2], outputs=x)
adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='mse')
hist = model.fit({'input1': echo1, 'input2': echo2}, {'imaging': imaging}, shuffle=True, epochs=10, batch_size=batch_size, validation_split=0.2, validation_data=None)
with open('loss1.txt', 'w') as f: f.write(str(hist.history))
model.save('CV_GMTINet_10epoch.h5')

adam = Adam(lr=5e-5)
model.compile(optimizer=adam, loss='mse')
hist = model.fit({'input1': echo1, 'input2': echo2}, {'imaging': imaging}, shuffle=True, epochs=10, batch_size=batch_size, validation_split=0.2, validation_data=None)
with open('loss2.txt', 'w') as f: f.write(str(hist.history))
model.save('CV_GMTINet_20epoch.h5')