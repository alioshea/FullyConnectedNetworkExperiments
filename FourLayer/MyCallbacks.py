'''
	My Callback functions described here
'''

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, warnings
from keras import backend as K
#from AUC_pool import calc_roc
from score_tool_DNN_resp import calc_roc
import numpy as np

window_size = 61
def movingaverage(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, "same")

class LearningRateDecay(Callback):
    '''
	reduces the learning rate every n_epochs

    '''
    def __init__(self, decay, n_epochs, verbose=0):
        Callback.__init__(self)
        self.decay = decay
        self.n_epochs = n_epochs
        self.verbose = verbose
    def on_epoch_end(self, epoch, logs={}):
        if not (epoch and epoch % n_epochs == 0):
            return
        assert hasattr(self.model.optimizer, 'lr'), \
            'Optimizer must have a "lr" attribute.'
        current_lr = K.get_value(self.model.optimizer.lr) 
        new_lr = current_lr*self.decay
        if self.verbose>0:
            print(' \nEpoch %05d: reducing learning rate' % (epoch))
            sys.stderr.write('new lr: %.5f\n\n' % new_lr)
        K.set_value(self.model.optimizer.lr, new_lr)
        

class SavingIntermediateModel(Callback):
    '''
        Saving the weights and learning curves every_n iterations.
        This is maybe a good idea to keep an eye on learning for long 
        experiments and to have intermediate weight values saved.

    '''
    def __init__(self, validation, train, test, every_n=25, verbose=0, folder_test ="", network_name = ""):
        Callback.__init__(self)
        self.every_n = every_n
        self.verbose = verbose
        self.network_name = network_name
        self.folder_test = folder_test
        self.validation = validation
        self.train = train
        self.test = test

    def on_epoch_end(self, epoch, logs={}):
        if not (epoch and epoch % self.every_n == 0):
            return
        
        location = './Data/32_'+str(self.folder_test)+'_'+str(self.network_name)+"_"+str(epoch)

        np.save(location + "_validation.npy", self.validation.AUC)
        np.save(location + "_train.npy", self.train.AUC)
        np.save(location + "_test.npy", self.test.AUC)

        self.model.save_weights(location + "_weights.hdf5")


class AUCCallback(Callback): 
    '''
        Providing an update of the AUC score for each dataset (train/test/validation) 
        at the end of each epoch. Can be used to decide weather the current model is the
        'best' model and if the current weights should be saved.

    '''

    def __init__(self, data=(), patience=15, saving_model_name='_finalAUC', batch_size=2048, location='', data_type='train'):
        super(Callback,self).__init__()
       # Callback.__init__(self)
        self.data_x, self.data_y = data
        self.patience = patience
        self.saving_model_name = saving_model_name
        self.batch_size = batch_size
        self.location = location
        self.data_type = data_type

        self.AUC = []
        self.best = -np.inf
        
    def on_epoch_end(self,epoch,logs={}):
        if epoch%2==0 or self.model.stop_training == True:		#Only calculate AUC if this condition is met
           # Pass the data though the network to get probabilistic predictions
           processed = []
           if self.data_type == 'train':
               p = self.model.predict(self.data_x)
               p = np.transpose(p)[1]
               processed.append(p)
               
           elif self.data_type == 'validation' or self.data_type == 'test':
               for files in self.data_x:
                   eight_channels = []
                   for channel in files:
                       channel = np.asarray(channel)
                       array = channel.reshape((channel.shape[0],channel.shape[1],1))
                       p = self.model.predict(array,batch_size=self.batch_size, verbose=0)
                       p = np.transpose(p)
                       eight_channels.append(movingaverage(p[1],window_size))
                   self.max = eight_channels[0]
                   for chan_num in range(7):
                       self.max = np.maximum(self.max,eight_channels[chan_num+1])
                   processed.append(self.max)
           # Get the ROC values of network outputs         
           auc_all = []
           x = []
           y = []
           for file_number in range(len(self.data_y)):
               y = np.reshape(np.array(self.data_y[file_number]).astype("int"),(len(self.data_y[file_number]))) 
               x = np.reshape(processed[file_number],(len(processed[file_number])))    
               auc_all.append(roc_auc_score(y,x))
           current_auc = np.average(auc_all)
           self.AUC.append(current_auc)
           if self.data_type == 'validation':
               print('\nEpoch %05d: Validation AUC %f' % (epoch + 1,current_auc*100))
               if current_auc > self.best:
                   self.best = current_auc
                   self.model.save_weights(self.saving_model_name,overwrite=True)
                   print('Epoch %05d: Best Weights' % (epoch + 1))
                   print('New Best Validation AUC: %f' % (current_auc*100))
               if epoch >=98:
                   self.model.stop_training = True
                   print('\nEpoch %05d: early stopping' % (epoch + 1))
                   self.model.load_weights(self.saving_model_name)
           if self.data_type == "test" or self.data_type == 'train':
               print('Epoch %05d: %s AUC %f\n' % (epoch + 1,self.data_type, current_auc*100))


           if self.data_type == "test" and  self.model.stop_training == True:
               self.model.load_weights(self.saving_model_name)
               for files in self.data_x:
                    eight_channels = []
                    for channel in files:
                        channel = np.asarray(channel)
                        array = channel.reshape((channel.shape[0],channel.shape[1],1))
                        p = self.model.predict(array,batch_size=self.batch_size, verbose=0)
                        p = np.transpose(p)
                        eight_channels.append(movingaverage(p[1],window_size))
               auc_pp = calc_roc(eight_channels,y)
               print("AUC and Collar Calculation on Best Model")
               np.save(self.saving_model_name[:-5]+'AUC_test',auc_pp)

