import numpy as np
import numpy
import torch, torchvision
from torch import nn, optim
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
import seaborn as sns
from sklearn.decomposition import PCA
from umap import UMAP
from package.umap.umap.parametric_umap import ParametricUMAP
import pickle
from umap.parametric_umap import load_ParametricUMAP
from sklearn import preprocessing
import matplotlib
import os
sns.set()

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[1],'GPU')

# load MNIST
mnist_train = torchvision.datasets.MNIST(root='./',
                                         train=True,
                                         download=True, 
                                         transform=None)
                                        #  transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]))
x_train, y_train = mnist_train.data.float().numpy(), mnist_train.targets

mnist_test = torchvision.datasets.MNIST(root='./',
                                        train=False,
                                        download=True, 
                                        transform=None)
                                        # transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]))
x_test, y_test = mnist_test.data.float().numpy(), mnist_test.targets

x_train2d = x_train.reshape((x_train.shape[0], -1))/255.
x_test2d = x_test.reshape((x_test.shape[0], -1))/255.

x2d = np.concatenate([x_train2d, x_test2d], axis=0)
# min_max_scaler = preprocessing.MinMaxScaler()
# x2d = min_max_scaler.fit_transform(x2d)

y = np.concatenate([y_train, y_test], axis=0)

def encoder_linear(input_dim, latent_dim):
    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_dim),
        tf.keras.layers.Activation("elu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation="elu"),
        tf.keras.layers.Dense(units=256, activation="elu"),
        tf.keras.layers.Dense(units=128, activation="elu"),
        tf.keras.layers.Dense(units=64, activation="elu"),
        tf.keras.layers.Dense(units=latent_dim, name="z")])
    return encoder

def decoder_linear(input_dim, latent_dim):
    decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=latent_dim),
        tf.keras.layers.Dense(units=64, activation="elu"),
        tf.keras.layers.Dense(units=128, activation="elu"),
        tf.keras.layers.Dense(units=256, activation="elu"),
        tf.keras.layers.Dense(units=512, activation="elu"),
        tf.keras.layers.Dense(units=input_dim, name="recon", activation=None)])
    return decoder

def encoder_linear_BN(input_dim, latent_dim):
    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_dim),
        tf.keras.layers.Activation("elu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation="elu", kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=256, activation="elu", kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=128, activation="elu", kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=64, activation="elu", kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=latent_dim, name="z")])
    return encoder

def decoder_linear_BN(input_dim, latent_dim):
    decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=latent_dim),
        tf.keras.layers.Dense(units=64, activation="elu", kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=128, activation="elu", kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=256, activation="elu", kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=512, activation="elu", kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=input_dim, name="recon", activation=None)])
    return decoder

def plot_embeddings(x_data,y_data,dim,folder,model,timing,plottype):
    sns.set_style("whitegrid")
    if x_data.shape[-1] == 2:
        plt.figure(dpi=200)
        plt.scatter(*x_data.T, c=y_data, alpha=0.5, s=1.0, cmap="tab10", edgecolor="none")
        plt.colorbar()
        plt.gca().set_aspect("equal")
        plt.axis("off")
        plt.title(model+str(x_data.shape[-1])+str(dim))
        plt.savefig(folder+model+str(dim)+timing+plottype+'_embedding.png', bbox_inches='tight')
    elif x_data.shape[-1] == 3:
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_data.T[0],x_data.T[1],x_data.T[2], c=y_data, alpha=0.5, s=1.0, cmap="tab10", edgecolor="none")
        plt.title(model+str(x_data.shape[-1])+str(dim))
        plt.savefig(folder+model+str(dim)+timing+plottype+'_embedding.png', bbox_inches='tight')
    else:
        pca = PCA(n_components=3)
        components = pca.fit_transform(x_data)
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(components.T[0],components.T[1],components.T[2], c=y_data, alpha=0.5, s=1.0, cmap="tab10", edgecolor="none")
        ax.set_title(model+str(x_data.shape[-1])+str(dim)+' PCA')
        plt.savefig(folder+model+str(dim)+timing+plottype+'_PCA_embedding.png', bbox_inches='tight')
    plt.close()
    return

def plot_embeddings_new(x_data,y_data,dim,folder,model,timing,plottype):
    sns.set_style("whitegrid")
    if x_data.shape[-1] == 2:
        plt.figure(dpi=200)
        plt.scatter(*x_data.T, c=y_data, alpha=0.5, s=1.0, cmap="seismic",vmax=max(y_data)/2,edgecolor="none")
        plt.colorbar()
        plt.gca().set_aspect("equal")
        plt.axis("off")
        plt.title(model+str(x_data.shape[-1])+str(dim)+'D')
        plt.savefig(folder+model+str(dim)+timing+plottype+'2_embedding.png', bbox_inches='tight')
    plt.close()
    if x_data.shape[-1] == 2:
        plt.figure(dpi=200)
        plt.scatter(*x_data.T, c=y_data, alpha=0.5, s=1.0, cmap="seismic",vmax=max(y_data)/3,edgecolor="none")
        plt.colorbar()
        plt.gca().set_aspect("equal")
        plt.axis("off")
        plt.title(model+str(x_data.shape[-1])+str(dim)+'D')
        plt.savefig(folder+model+str(dim)+timing+plottype+'3_embedding.png', bbox_inches='tight')
    plt.close()

def plot_reconstructions(data,recon,dim,folder,model,plottype):
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(data[i].reshape(28,28))
        plt.title(model+"original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recon[i].reshape(28,28))
        plt.title(model+"reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(folder+model+str(dim)+plottype+'_reconstructions.png', bbox_inches='tight')
    plt.close()
    return

def plot_umap_ae_loss(data,dim,folder,model):
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(2,1)
    ax[0].plot(data[1])
    ax[1].plot(data[2])
    ax[0].plot([], [], ' ', label='final loss '+str(round(data[1][-1],2)))
    ax[1].plot([], [], ' ', label='final loss '+str(round(data[2][-1],2)))
    ax[0].set_ylabel('MSE Loss')
    ax[1].set_ylabel('UMAP Loss')
    ax[0].set_title(model+str(dim)+' Reconstruction Loss and UMAP Loss')
    ax[1].set_xlabel('Epoch')
    ax[0].legend()
    ax[1].legend()
    plt.savefig(folder+model+str(dim)+'_loss.png', bbox_inches='tight')
    plt.close()
    return

def plot_loss(data,dim,folder,model):
    sns.set_style("darkgrid")
    mean_loss = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    plt.errorbar(range(len(mean_loss)), mean_loss, yerr=std, capsize=5, marker='o',label='loss')
    plt.plot([], [], ' ', label='final mean loss '+str(round(mean_loss[-1],2)))
    plt.plot([], [], ' ', label='final abs loss '+str(round(data[-1,-1],2)))
    plt.title(model+str(dim)+'Average loss per epoch (± std)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(folder+model+str(dim)+'_loss.png', bbox_inches='tight')
    plt.close()
    return

def umap(data,trainX,testX,trainY,testY,dim):
    folder = './umap_wdbn/'
    model = 'umap_BN'
    input_dim = data.shape[-1]
    dim = dim
    #train umap
    embedder = UMAP(n_components=dim)
    embedding = embedder.fit_transform(x2d)
    #save embeddings
    np.savetxt(folder+model+str(dim)+'D.csv',embedding, delimiter=',')
    #load embedding
    embedding = np.genfromtxt(folder+model+str(dim)+'D.csv', delimiter=',')
    #test train split
    embedding_train = embedding[:60000]
    embedding_test = embedding[60000:]
    #plot embeddings
    plot_embeddings(embedding_train,y_train,dim,folder,model)
    #train decoder
    latent_dim = embedding_train.shape[-1]
    decoder = decoder_linear_BN(input_dim, latent_dim)
    dec_model = Model(inputs=decoder.input, outputs=decoder.output)
    dec_model.compile(optimizer='adam',loss='mse')
    history_l = []
    for i in range(5):
        checkpoint_filepath = folder+'checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='loss',
            mode='max',
            save_best_only=True)
        #fit model
        dec_history = dec_model.fit(embedding_train,trainX,batch_size=256,epochs=40,callbacks=[model_checkpoint_callback])
        history_l.append(dec_history)
    #save history
    losses = [h.history['loss'] for h in history_l]
    np.savetxt(folder+model+str(latent_dim)+'D_history.csv', losses, delimiter=',')
    #load history
    losses = np.genfromtxt(folder+model+str(latent_dim)+'D_history.csv', delimiter=',')
    #reconstructions
    decoded_imgs = decoder(embedding_test).numpy()
    #plot reconstructions
    plot_reconstructions(x_test,decoded_imgs,dim,folder,model)
    #plot loss
    plot_loss(losses,dim,folder,model)
    return

def umap_AE(data,dim):
    folder = './umap_ae_wdbn/'
    model = 'umap_AE_BN'
    input_dim = data.shape[-1]
    latent_dim = dim
    #train umap_AE
    embedder = ParametricUMAP(encoder=encoder_linear_BN(input_dim, latent_dim),
                          decoder=decoder_linear_BN(input_dim, latent_dim),
                          autoencoder_loss=True,parametric_reconstruction=True,n_components=latent_dim,
                          parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
                          verbose=True,
                          parametric_reconstruction_loss_weight=1.0,
                          n_training_epochs=5,
                         )
    embedding = embedder.fit_transform(data)
    #save model and history
    pickle.dump(embedding, open(str(folder)+model+str(latent_dim)+'D.sav', 'wb'))
    np.savetxt(str(folder)+model+str(latent_dim)+'D.csv',[embedder._history['loss'],embedder._history['reconstruction_loss'],embedder._history['umap_loss']], delimiter=',')
    #load model and history
    embedder1 = np.genfromtxt(str(folder)+model+str(latent_dim)+'D.csv', delimiter=',')
    embedding = pickle.load((open(str(folder)+model+str(latent_dim)+'D.sav', 'rb')))
    #train test split
    embedding_train = embedding[:60000]
    embedding_test = embedding[60000:]
    #plot embeddings
    plot_embeddings(embedding_train,y_train,dim,folder,model)
    #reconstructions
    reconstruction = embedder.inverse_transform(embedding_test)
    #plot reconstructions
    plot_reconstructions(x_test,reconstruction,dim,folder,model)
    #plot loss
    plot_umap_ae_loss(embedder1,dim,folder,model)
    return

def vanilla_AE(trainX,testX,trainY,testY,dim):
    folder = './vanilla_ae_wdbn/'
    model = 'vanilla_AE_BN'
    input_dim = trainX.shape[-1]
    latent_dim = dim
    #train vanilla AE
    encoder = encoder_linear_BN(input_dim, latent_dim)
    decoder = decoder_linear_BN(input_dim, latent_dim)
    #compile model
    AE_model = Model(inputs=encoder.input, outputs=decoder(encoder.output))
    AE_model.compile(optimizer='adam',loss='mse')
    #training loop
    history_l = []
    for i in range(5):
        checkpoint_filepath = folder+'checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='loss',
            mode='max',
            save_best_only=True)
        #fit model
        history = AE_model.fit(trainX,trainX,batch_size=256,epochs=40,callbacks=[model_checkpoint_callback])
        history_l.append(history)
    #save history
    losses = [h.history['loss'] for h in history_l]
    np.savetxt(folder+model+str(latent_dim)+'D_history.csv', losses, delimiter=',')
    #load history
    losses = np.genfromtxt(folder+model+str(latent_dim)+'D_history.csv', delimiter=',')
    #test embeddings
    encoded_imgs = encoder(testX).numpy()
    #test reconstruction
    decoded_imgs = decoder(encoded_imgs).numpy()
    #plot embeddings
    plot_embeddings(encoded_imgs,y_test,dim,folder,model)
    #plot reconstructions
    plot_reconstructions(x_test,decoded_imgs,dim,folder,model)
    #plot loss
    plot_loss(losses,dim,folder,model)
    return

# for i in [2,4,8,16,32,64,128,256]:
#     umap(x2d,x_train2d,x_test2d,y_train,y_test,i)
#     umap_AE(x2d,i)
#     vanilla_AE(x_train2d,x_test2d,y_train,y_test,i)

#change one weight either MSE or UMAP loss
def umap_AE_umap_loss_weight(data,dim,weight,folder):
    model = 'umap_AE'
    input_dim = data.shape[-1]
    latent_dim = dim[0]
    #train umap_AE
    embedder = ParametricUMAP(encoder=encoder_linear_BN(input_dim, latent_dim),
                          decoder=decoder_linear_BN(input_dim, latent_dim),
                          autoencoder_loss=True,parametric_reconstruction=True,n_components=latent_dim,
                          parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
                          verbose=True,
                          parametric_reconstruction_loss_weight=1.0,umap_loss_weight = weight,
                          n_training_epochs=5,
                         )
    embedding = embedder.fit_transform(data)
    #save model and history
    pickle.dump(embedding, open(str(folder)+model+str(latent_dim)+'D'+str(weight)+'.sav', 'wb'))
    np.savetxt(str(folder)+model+str(latent_dim)+'D'+str(weight)+'.csv',[embedder._history['loss'],embedder._history['reconstruction_loss'],embedder._history['umap_loss']], delimiter=',')
    #load model and history
    embedder1 = np.genfromtxt(str(folder)+model+str(latent_dim)+'D'+str(weight)+'.csv', delimiter=',')
    embedding = pickle.load((open(str(folder)+model+str(latent_dim)+'D'+str(weight)+'.sav', 'rb')))
    #train test split
    embedding_train = embedding[:60000]
    embedding_test = embedding[60000:]
    #plot embeddings
    plot_embeddings(embedding_train,y_train,dim,folder,model)
    #reconstructions
    reconstruction = embedder.inverse_transform(embedding_test)
    #plot reconstructions
    plot_reconstructions(x_test,reconstruction,dim,folder,model)
    #plot loss
    plot_umap_ae_loss(embedder1,dim,folder,model)
    return

# folders = ['./umap_ae_loss_weight_umap/','./umap_ae_loss_weight_umap1/']
# weights = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e+1,1e+2,1e+3,1e+4,1e+5,1e+6,1e+7,1e+8]
# for folder in folders:
#     for i in weights:
#         umap_AE_umap_loss_weight(x2d,[2,i],i,folder)

def umap_AE_rec_loss_weight(data,dim,weight,folder):
    model = 'umap_AE'
    input_dim = data.shape[-1]
    latent_dim = dim[0]
    #train umap_AE
    embedder = ParametricUMAP(encoder=encoder_linear_BN(input_dim, latent_dim),
                          decoder=decoder_linear_BN(input_dim, latent_dim),
                          autoencoder_loss=True,parametric_reconstruction=True,n_components=latent_dim,
                          parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
                          verbose=True,
                          parametric_reconstruction_loss_weight=weight,umap_loss_weight = 1.0,
                          n_training_epochs=5,
                         )
    embedding = embedder.fit_transform(data)
    #save model and history
    pickle.dump(embedding, open(str(folder)+model+str(latent_dim)+'D'+str(weight)+'.sav', 'wb'))
    np.savetxt(str(folder)+model+str(latent_dim)+'D'+str(weight)+'.csv',[embedder._history['loss'],embedder._history['reconstruction_loss'],embedder._history['umap_loss']], delimiter=',')
    #load model and history
    embedder1 = np.genfromtxt(str(folder)+model+str(latent_dim)+'D'+str(weight)+'.csv', delimiter=',')
    embedding = pickle.load((open(str(folder)+model+str(latent_dim)+'D'+str(weight)+'.sav', 'rb')))
    #train test split
    embedding_train = embedding[:60000]
    embedding_test = embedding[60000:]
    #plot embeddings
    plot_embeddings(embedding_train,y_train,dim,folder,model)
    #reconstructions
    reconstruction = embedder.inverse_transform(embedding_test)
    #plot reconstructions
    plot_reconstructions(x_test,reconstruction,dim,folder,model)
    #plot loss
    plot_umap_ae_loss(embedder1,dim,folder,model)
    return

# folders = ['./umap_ae_loss_weight_rec/','./umap_ae_loss_weight_rec1/']
# weights = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e+1,1e+2,1e+3,1e+4,1e+5,1e+6,1e+7,1e+8]
# for folder in folders:
#     for i in weights:
#         umap_AE_rec_loss_weight(x2d,[2,i],i,folder)

# double train 10 epochs with umap loss then no umap loss
def double_train_loss_weight(data,dim):
    weight1 = dim[1]
    weight2 = dim[2]
    run = dim[3]
    epoch1 = dim[4]
    epoch2 = dim[5]
    
    folder = './double_train_weight_umap3/'+str(epoch1)+'_'+str(epoch2)+'/'
    os.makedirs(folder, exist_ok=True)
    model = 'umap_AE_run_'+str(run)+'_'
    input_dim = data.shape[-1]
    latent_dim = dim[0]
    
    before = 'before_'
    after = 'after_'
    final = 'final_'
    
    train_data = data[:60000]
    test_data = data[60000:]
    
    #train umap_AE
    embedder = ParametricUMAP(batch_size=1000, encoder=encoder_linear_BN(input_dim, latent_dim),
                          decoder=decoder_linear_BN(input_dim, latent_dim),
                          autoencoder_loss=True,parametric_reconstruction=True,n_components=latent_dim,
                          parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
                          verbose=True,
                          parametric_reconstruction_loss_weight=1,umap_loss_weight=weight1,
                          n_training_epochs=epoch1,loss_report_frequency=1
                         )
    #fit once
    embedding = embedder.fit_transform(train_data)
    #save model once
    pickle.dump(embedding, open(folder+model+str(dim)+before+'.sav', 'wb'))
#     embedder.save(folder+model+str(dim)+before)
    #train test split
    embedding_train = embedding
    embedding_test = embedder.transform(test_data)
    #plot embeddings
    plot_embeddings(embedding_train,y_train,dim,folder,model,before,'train')
    plot_embeddings(embedding_test,y_test,dim,folder,model,before,'test')
    first_loss = [embedder._history['loss'],embedder._history['reconstruction_loss'],embedder._history['umap_loss']]
    #####################################################################################################################
    #reconstructions
    reconstruction_train = embedder.inverse_transform(embedding_train)
    reconstruction_test = embedder.inverse_transform(embedding_test)
    #save reconstructions
    reconstruction_all = np.concatenate((reconstruction_train,reconstruction_test))
    pickle.dump(reconstruction_all, open(folder+model+str(dim)+before+'reconstructions.sav', 'wb'))
    #####################################################################################################################
    #change params
    embedder.umap_loss_weight = weight2
    embedder.n_training_epochs = 1
    #fit twice
    embedding = embedder.fit_transform(train_data)
    #save model twice
    pickle.dump(embedding, open(folder+model+str(dim)+after+'.sav', 'wb'))
#     embedder.save(folder+model+str(dim)+after)
    #train test split
    embedding_train = embedding
    embedding_test = embedder.transform(test_data)
    #plot embeddings
    plot_embeddings(embedding_train,y_train,dim,folder,model,after,'train')
    plot_embeddings(embedding_test,y_test,dim,folder,model,after,'test')
    second_loss = [embedder._history['loss'],embedder._history['reconstruction_loss'],embedder._history['umap_loss']]
    #####################################################################################################################
    #reconstructions
    reconstruction_train = embedder.inverse_transform(embedding_train)
    reconstruction_test = embedder.inverse_transform(embedding_test)
    #save reconstructions
    reconstruction_all = np.concatenate((reconstruction_train,reconstruction_test))
    pickle.dump(reconstruction_all, open(folder+model+str(dim)+after+'reconstructions.sav', 'wb'))
    #####################################################################################################################
    #change params
    embedder.umap_loss_weight = weight2
    embedder.n_training_epochs = epoch2
    #fit thrice
    embedding = embedder.fit_transform(train_data)
    third_loss = [embedder._history['loss'],embedder._history['reconstruction_loss'],embedder._history['umap_loss']]
    total_loss = np.concatenate((first_loss,second_loss,third_loss),axis=1)
    #save model and history
    pickle.dump(embedding, open(folder+model+str(dim)+final+'.sav', 'wb'))
#     embedder.save(folder+model+str(dim)+final)
    np.savetxt(folder+model+str(dim)+'history.csv',total_loss, delimiter=',')
    #load embedding and history
    history = np.genfromtxt(folder+model+str(dim)+'history.csv', delimiter=',')
    embedding = pickle.load((open(folder+model+str(dim)+final+'.sav', 'rb')))
    #train test split
    embedding_train = embedding
    embedding_test = embedder.transform(test_data)
    #plot embeddings
    plot_embeddings(embedding_train,y_train,dim,folder,model,final,'train')
    plot_embeddings(embedding_test,y_test,dim,folder,model,final,'test')
    #####################################################################################################################
    #reconstructions
    reconstruction_train = embedder.inverse_transform(embedding_train)
    reconstruction_test = embedder.inverse_transform(embedding_test)
    #save reconstructions
    reconstruction_all = np.concatenate((reconstruction_train,reconstruction_test))
    pickle.dump(reconstruction_all, open(folder+model+str(dim)+final+'reconstructions.sav', 'wb'))
    #####################################################################################################################
    #plot reconstructions
    plot_reconstructions(x_train2d,reconstruction_train,dim,folder,model,'train')
    plot_reconstructions(x_test2d,reconstruction_test,dim,folder,model,'test')
    #calculate mse and save
    train_mse = ((x_train2d-reconstruction_train)**2).mean(axis=1)
    test_mse = ((x_test2d-reconstruction_test)**2).mean(axis=1)
    np.savetxt(folder+model+str(dim)+'train_mse.csv',train_mse, delimiter=',')
    np.savetxt(folder+model+str(dim)+'test_mse.csv',train_mse, delimiter=',')
    #save mse embeddings
    plot_embeddings_new(embedding_train,train_mse,dim,folder,model,final,'train_mse')
    plot_embeddings_new(embedding_test,test_mse,dim,folder,model,final,'test_mse')
    #plot loss
    plot_umap_ae_loss(history,dim,folder,model)
    return

# weights_1 = [1,1,0,0]
# weights_2 = [1,0,0,1]

# epoch_1 = [1, 2, 3]
# epoch_2 = [28,27,26]

# for p in range(2):
#     for k,l in zip(epoch_1,epoch_2):
#         for i,j in zip(weights_1,weights_2):
#             print(i,j,k,l)
#             double_train_loss_weight(x2d,[2,i,j,p,k,l])

# double train 10 epochs with umap loss then no umap loss
def double_train_loss_weight(data,dim):
    weight1 = dim[1]
    weight2 = dim[2]
    run = dim[3]
    epoch1 = dim[4]
    epoch2 = dim[5]
    
    folder = './testing123/'+str(epoch1)+'_'+str(epoch2)+'/'
    os.makedirs(folder, exist_ok=True)
    model = 'umap_AE_run_'+str(run)+'_'
    input_dim = data.shape[-1]
    latent_dim = dim[0]
    
    train_data = data[:60000]
    test_data = data[60000:]
    
    model_loss = []
    model_mse = []
    model_umap = []
    tensorflow_loss = []
    numpy_loss = []
    
    #train umap_AE
    embedder = ParametricUMAP(batch_size=1000, encoder=encoder_linear_BN(input_dim, latent_dim),
                          decoder=decoder_linear_BN(input_dim, latent_dim),
                          autoencoder_loss=True,parametric_reconstruction=True,n_components=latent_dim,
                          parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
                          verbose=True,
                          parametric_reconstruction_loss_weight=1,umap_loss_weight=0,
                          n_training_epochs=1,loss_report_frequency=1
                         )
    for i in range(20):
        #fit
        embedding = embedder.fit_transform(train_data)
        #append model mse
        model_loss.append(embedder._history['loss'][0])
        model_mse.append(embedder._history['reconstruction_loss'][0])
        model_umap.append(embedder._history['umap_loss'][0])
        
        #train test split
        embedding_train = embedding
        embedding_test = embedder.transform(test_data)
        
        #reconstructions
        reconstruction_train = embedder.inverse_transform(embedding_train)
        reconstruction_test = embedder.inverse_transform(embedding_test)

        #calculate numpy mse
        train_mse = np.mean(np.square(x_train2d-reconstruction_train),axis=1)
        test_mse = np.mean(np.square(x_test2d-reconstruction_test),axis=1)
        
        #hist numpy
        counts1,bin_loc1,_ = plt.hist(train_mse,bins=1000)
        numpy_loss.append((counts1*bin_loc1[:-1]).sum()/len(x_train2d))
        plt.close()
        
        #calculate tensorflow mse
        mse_tf = tf.keras.losses.MeanSquaredError()
        tensorflow_loss.append(mse_tf(x_train2d, reconstruction_train).numpy())
        
    np.savetxt(folder+model+str(dim)+'model_loss.csv',model_loss, delimiter=',')
    np.savetxt(folder+model+str(dim)+'model_mse.csv',model_mse, delimiter=',')
    np.savetxt(folder+model+str(dim)+'model_umap.csv',model_umap, delimiter=',')
    np.savetxt(folder+model+str(dim)+'numpy_loss.csv',numpy_loss, delimiter=',')
    np.savetxt(folder+model+str(dim)+'tensorflow_loss.csv',tensorflow_loss, delimiter=',')

double_train_loss_weight(x2d,[2,0,0,0,0,20])