import numpy as np
import tensorflow as tf
import math
import time
from sklearn import metrics
import matplotlib.pyplot as plt
start_time = time.time()

def embed_diagonal(inp):
    Batches=[]
    for batch in inp:
        diag_matrix=tf.linalg.diag(batch)
        Batches.append(diag_matrix)
    output=tf.stack(Batches)

    return output

def get_loss(y_true, y_pred):
    # No loss on diagonal

    datatype=tf.float32.max
    dimension=y_pred.get_shape().as_list()[-1]
    List=[]
    for i in y_pred:   
        List.append(tf.linalg.set_diag(i, [datatype]*dimension))
       
    y_pred=tf.convert_to_tensor(List)
    

    #calculate loss
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=(True))(y_true,y_pred)  # cross entropy 
    y_pred = tf.math.sigmoid(y_pred)
    tp = tf.math.reduce_sum(tf.multiply(y_pred, y_true),[1,2])
    fn = tf.math.reduce_sum((y_true - tf.multiply(y_pred, y_true)),[1,2])
    fp = tf.math.reduce_sum((y_pred -tf.multiply(y_pred,y_true)),[1,2])
    loss = loss - ((2 * tp) / tf.math.reduce_sum((2 * tp + fp + fn + 1e-10)))  # fscore

    return loss

class Attention(tf.keras.Model):
    def __init__(self, in_features):
        super(Attention, self).__init__()
        
        small_in_features = max(math.floor(in_features/10), 1)
        self.d_k = small_in_features

        self.query=(tf.keras.layers.Dense(small_in_features,activation="tanh",name="query"))
        self.key = tf.keras.layers.Dense(small_in_features,name="key")

    def call(self, inp):
        # inp.shape should be (B,N,C)
        q = self.query(inp)  # (B,N,C/10)
        k = self.key(inp)     # B,N,C/10
        #k = tf.transpose(k,perm=[0,2,1])
        k= tf.linalg.matrix_transpose(k)
        x = tf.linalg.matmul(q, k) / math.sqrt(self.d_k)  # B,N,N

        x = tf.nn.softmax(x,axis=2)  # over rows

        x = tf.linalg.matmul(x, inp)  # (B, N, C)

        return x

class DiagOffdiagMLP(tf.keras.Model):
    def __init__(self, in_features, out_features, seperate_diag):
        super(DiagOffdiagMLP, self).__init__()
        self.seperate_diag = seperate_diag
        self.conv_offdiag = tf.keras.layers.Conv2D(out_features,kernel_size=1,name="conv2d_diag")
        if self.seperate_diag:
            self.conv_diag = tf.keras.layers.Conv1D(out_features, kernel_size=1,name="con1d_diag")
    def call(self, inp):
        # Assume x.shape == (B, C, N, N)
        inp_diag=tf.linalg.diag_part(inp)
        if self.seperate_diag:
            return self.conv_offdiag(inp) + embed_diagonal(self.conv_diag(inp_diag))
        return self.conv_offdiag(inp)

class DeepSetLayer(tf.keras.Model):
    def __init__(self, in_features, out_features, attention, normalization, second_bias):
        """
        DeepSets single layer
        :param input:shape: input's shape
        :param out_features: output's number of features
        :param attention: Whether to use attention
        :param normalization: normalization method - 'fro' or 'batchnorm'
        :param second_bias: use a bias in second conv1d layer
        """
        super(DeepSetLayer, self).__init__()
        self.attention = attention
        if attention:
            self.Attention = Attention(in_features=in_features)
        self.layer1 = tf.keras.layers.Conv1D(out_features, kernel_size=1)
        self.layer2 = tf.keras.layers.Conv1D(out_features, kernel_size=1, use_bias=second_bias)

        self.normalization = normalization
        if normalization == 'batchnorm':
            self.bn = tf.keras.layers.BatchNormalization(out_features)

    def call(self, x):
        #tf.shape(x) = (batch_size,variables,n_tracks)
        # attention
        if self.attention:
            #x_T = tf.transpose(x,perm=[0,2,1]) #(batch_size,variables,n_tracks)->(batch_size,n_tracks,variables)     
            x_T= tf.linalg.matrix_transpose(x)
            x = self.layer1(x_T) + self.layer2(self.Attention(x_T))
            #x= tf.transpose(x,perm=[0,2,1])
            x= tf.linalg.matrix_transpose(x)
        else:
            x = self.layer1(x) + self.layer2(x - tf.math.reduce_mean(x,axis=2,keepdims=True))

        # normalization
        if self.normalization == 'batchnorm':
            x = self.bn(x)
        else:
            x = x / tf.norm(x,ord="fro" ,axis=[1,2], keepdims=True)  # BxCxN / Bx1xN
        return x

class PsiSuffix(tf.keras.Model):
    def __init__(self, features, predict_diagonal):
        super(PsiSuffix,self).__init__()
        layers = []
        for i in range(len(features) - 2):
            layers.append(DiagOffdiagMLP(features[i], features[i + 1], predict_diagonal))
            layers.append(tf.keras.layers.ReLU())
        layers.append(DiagOffdiagMLP(features[-2], features[-1], predict_diagonal))
        self.model = tf.keras.models.Sequential(layers)

    def call(self, x):
        return self.model(x)

class DeepSet(tf.keras.Model):
    def __init__(self, in_features, feats, attention, cfg=None):
        """
        DeepSets implementation
        :param in_features: input's number of features
        :param feats: list of features for each deepsets layer
        :param attention: True/False to use attention
        :param cfg: configurations of second_bias and normalization method
        """
        super(DeepSet, self).__init__()
        
        layers = []
        normalization =  'fro'
        second_bias =  True

        layers.append(DeepSetLayer(in_features, feats[0], attention, normalization, second_bias))
        for i in range(1, len(feats)):
            layers.append(tf.keras.layers.ReLU())
            layers.append(DeepSetLayer(feats[i-1], feats[i], attention, normalization, second_bias))

        self.sequential = tf.keras.models.Sequential(layers)

    def call(self, x):
        return self.sequential(x)

class SetToGraph(tf.keras.Model):
    def __init__(self, in_features, out_features, set_fn_feats, method, hidden_mlp, predict_diagonal, attention):
        """
        SetToGraph model.
        :param in_features: input set's number of features per data point
        :param out_features: number of output features.
        :param set_fn_feats: list of number of features for the output of each deepsets layer
        :param method: transformer method - quad, lin2 or lin5
        :param hidden_mlp: list[int], number of features in hidden layers mlp.
        :param predict_diagonal: Bool. True to predict the diagonal (diagonal needs a separate psi function).
        :param attention: Bool. Use attention in DeepSets
        :param cfg: configurations of using second bias in DeepSetLayer, normalization method and aggregation for lin5.
        """
        super(SetToGraph, self).__init__()
        assert method in ['lin2', 'lin5']
        self.method = method

        self.agg =  tf.math.reduce_sum

        self.set_model = DeepSet(in_features, feats=set_fn_feats, attention=attention)

        # Suffix - from last number of features, to 1 feature per entrance
        d2 = (2 if method == 'lin2' else 5) * set_fn_feats[-1]
        hidden_mlp = [d2] + hidden_mlp + [out_features]
        self.suffix = PsiSuffix(hidden_mlp, predict_diagonal=predict_diagonal)
    

    def call(self, x):  
        #x=tf.transpose(x,perm=[0,2,1]) #from (batch_size,n_tracks,variables) to (batch_size,variables,n_tracks)
        x=tf.linalg.matrix_transpose(x)
        u = self.set_model(x) #(batch_size,n_tracks,variables)
        n = u.get_shape().as_list()[2]
        if self.method == 'lin2':
            m1 = tf.tile(tf.expand_dims(u,2),[1, 1, n, 1])  # broadcast to rows
            m2 = tf.tile(tf.expand_dims(u,3),[1, 1, 1, n]) # broadcast to cols
            block = tf.concat([m1, m2], axis=1)
            
        elif self.method == 'lin5':
            m1 = tf.tile(tf.expand_dims(u,2),[1, 1, n, 1])  # broadcast to rows
            m2 = tf.tile(tf.expand_dims(u,3),[1, 1, 1, n])  # broadcast to cols
            m3 = tf.tile(tf.expand_dims(self.agg(u, axis=2, keepdims=True),3),[1,1,n,n])  # sum over N, put on all
            m4 = embed_diagonal(u)  # assign values to diag only
            m5 = embed_diagonal(tf.tile(tf.expand_dims(self.agg(u, axis=2, keepdims=True),3),[1,1,n]))  # sum over N, put on diag
            block = tf.concat([m1, m2, m3, m4, m5], axis=1)
        
        block= tf.transpose(block,perm=[0,2,3,1])
        edge_vals = self.suffix(block)  # shape (B,N,N,out_features)
        edge_vals = tf.squeeze(edge_vals,axis=3)
        return edge_vals

phi=tf.keras.models.Sequential()
phi.add(SetToGraph(in_features=9,out_features=1,
                set_fn_feats=[256,256, 256,256, 5],
                method='lin2',
                hidden_mlp=[256],
                predict_diagonal=False,
                attention=True))


phi.compile(optimizer="adam",metrics=["accuracy"],loss=get_loss,run_eagerly=True )




validation=np.load("validation.npy",allow_pickle=True)
training=np.load("training.npy",allow_pickle=True)

for f in range(len(validation)-1,4,-1):

    validations=validation[f]
    trainings=training[f]
    phi.fit(trainings, validations,batch_size=512,epochs=4)

x=validation[35][0:12]
test=phi.predict(training[35][0:12])
test=tf.math.sigmoid(test)
test=test.numpy()
phi.save("jetmodel_weights")

a=(test.flatten())
b=np.array(x.flatten(),dtype=bool)
fpr,tpr,threshholds= metrics.roc_curve( b, a)
plt.plot(fpr,tpr)
print("--- %s seconds ---" % (time.time() - start_time))



