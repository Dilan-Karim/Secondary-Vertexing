import numpy as np
import tensorflow as tf
import math
import h5py


def embed_diagonal1(inp):
    Batches=[]
    for batch in inp:
        diag_matrix=tf.linalg.diag(batch)
        Batches.append(diag_matrix)
    output=tf.stack(Batches)
    print(tf.shape(output))
    return output

def get_loss(y_hat, y):
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_hat,y)  
    
    y_hat = tf.math.sigmoid(y_hat)
    
    tp = tf.math.reduce_sum(tf.multiply(y_hat, y),[1,2])
    fn = tf.math.reduce_sum((y - tf.multiply(y_hat, y)),[1,2])
    fp = tf.math.reduce_sum((y_hat -tf.multiply(y_hat,y)),[1,2])
    loss = loss - ((2 * tp) / tf.math.reduce_sum((2 * tp + fp + fn + 1e-10)))  # fscore

    return loss

class Attention(tf.keras.Model):
    def __init__(self, in_features):
        super(Attention, self).__init__()
        
        small_in_features = max(math.floor(in_features/10), 1)
        self.d_k = small_in_features

        #query = tf.keras.models.Sequential()
        #query.add(tf.keras.layers.Input())
        self.query=(tf.keras.layers.Dense(small_in_features,use_bias=True,trainable=True,activation="tanh",name="query"))
        self.key = tf.keras.layers.Dense(small_in_features,use_bias=True,trainable=True,name="key")

    def call(self, inp):
        # inp.shape should be (B,N,C)
        q = self.query(inp)  # (B,N,C/10)
        k = self.key(inp)     # B,N,C/10
        k = tf.transpose(k,perm=[0,2,1])
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
            return self.conv_offdiag(inp) + embed_diagonal1(self.conv_diag(inp_diag))
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
            self.Attention = Attention(in_features)
        self.layer1 = tf.keras.layers.Conv1D(out_features, kernel_size=1)
        self.layer2 = tf.keras.layers.Conv1D(out_features, kernel_size=1, use_bias=second_bias)

        self.normalization = normalization
        if normalization == 'batchnorm':
            self.bn = tf.keras.layers.BatchNormalization(out_features)

    def call(self, x):
        #tf.shape(x) = (B,C,N)
        # attention
        
        if self.attention:
            x_T = tf.transpose(x,perm=[0,2,1])  # B,C,N -> B,N,C
            x = self.layer1(x) + self.layer2(tf.transpose(self.Attention(x_T),perm=[0,2,1]))
            
        else:
            x = self.layer1(x) + self.layer2(x - tf.math.reduce_mean(x,axis=2,keepdims=True))
        

        # normalization
        if self.normalization == 'batchnorm':
            x = self.bn(x)
        else:
            x=tf.transpose(x,perm=[0,1,2])
            print(x)
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
        
        layers = [tf.keras.layers.InputLayer((256,9),name="Input_Deepset")]
        #layers=[]
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
        #x=tf.transpose(x,perm=[0,2,1]) # from BxNxC to BxCxN
        u = self.set_model(x)  # BxNx(out_features)
        u =tf.transpose(u,perm=[0,2,1]) # Bx(out_features)xN
        n = tf.shape(u)[2]
        if self.method == 'lin2':

            m1 = tf.tile(tf.expand_dims(u,2),[1, 1, n, 1])  # broadcast to rows
            m2 = tf.tile(tf.expand_dims(u,3),[1, 1, 1, n]) # broadcast to cols
            block = tf.concat([m1, m2], axis=1)
            
        elif self.method == 'lin5':
            m1 = tf.tile(tf.expand_dims(u,2),[1, 1, n, 1])  # broadcast to rows
            m2 = tf.tile(tf.expand_dims(u,3),[1, 1, 1, n])  # broadcast to cols
            m3 = tf.tile(tf.expand_dims(self.agg(u, axis=2, keepdims=True),3),[1,1,n,n])  # sum over N, put on all
            m4 = embed_diagonal1(u)  # assign values to diag only
            m5 = embed_diagonal1(tf.tile(tf.expand_dims(self.agg(u, axis=2, keepdims=True),3),[1,1,n]))  # sum over N, put on diag
            block = tf.concat([m1, m2, m3, m4, m5], axis=1)
        
        block= tf.transpose(block,perm=[0,2,3,1])
        edge_vals = self.suffix(block)  # shape (B,N,N,out_features)
        edge_vals = tf.squeeze(edge_vals,axis=3)
        return edge_vals

phi=tf.keras.models.Sequential()
phi.add(tf.keras.layers.InputLayer((256,9)))
phi.add(SetToGraph(in_features=9,out_features=1,
                set_fn_feats=[256,256, 256,256, 5],
                method='lin2',
                hidden_mlp=[256],
                predict_diagonal=False,
                attention=True))
phi.compile(optimizer="adam",metrics=["accuracy"],loss=get_loss)

phi.summary()
History=[]
for n in range(1,150):
    inputdata=h5py.File("IN"+str(n), "r").get("Tree")[()].astype(np.float16)
    outputdata=h5py.File("OUT"+str(n), "r").get("Tree")[()].astype(np.float16)
    callbacks=[]
    phi.fit(inputdata, outputdata,epochs=5,batch_size=75,callbacks=[callbacks])
    
    phi.save("trainedmodel_weights")
    History.append(callbacks)