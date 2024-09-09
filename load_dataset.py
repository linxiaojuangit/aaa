
'''
all indexes correspond to the date of fct with leadday 1
'''

import numpy as np
import torch
import h5py as h5
import torch.utils.data as data 
from config_args import get_args
import pickle as pkl

args = get_args()

class load_dataset( data.Dataset ):  #三个函数逐个加载数据  7+7

    def __init__( self, mode = None ):

        self.fname = '../data/data_samp/icec_samples.h5'

        if mode == 'train':
            self.grpname = 'trn' 
        elif mode == 'valid':
            self.grpname = 'vld'

    def __len__(self):  #样本数
        smp_len = 0
        with h5.File(self.fname, 'r') as f:
            for varname in f.keys():  #返回f中字典的键
                if varname.startswith('%s_clm_' % self.grpname):
                    smp_len = smp_len + 1
        return smp_len

    def __getitem__( self, index ):   #index是每个样本位置

        with h5.File(self.fname, 'r') as f:
            x_seq = []
            y_seq = []
            c_seq = []
            key = '%s_%04d/X' % (self.grpname, index)  #%s是样本编码 #百分号是格式化输出，这里说明两个百分号后面的内容
            x_seq.extend(f[key][:])

            key = '%s_%04d/C' % (self.grpname, index)
            c_seq.extend(f[key][:])

            key = '%s_%04d/Y' % (self.grpname, index)
            y_seq.extend(f[key][:])

        x = np.array( x_seq, dtype = np.float32)
        y = np.array( y_seq, dtype = np.float32)
        c = np.array( c_seq, dtype = np.float32)
        # x.shape: [ InLen * 2, width, height ]
        # y.shape: [ OutLen, width, height ]

        # Normalization
        # x = x / 100
        # y = y / 100
        # c = c / 100
        x[np.isnan(x)] = -1. #把空值设置成负一
        y[np.isnan(y)] = -1.
        c[np.isnan(c)] = -1.
        return torch.from_numpy(x), torch.from_numpy(y),torch.from_numpy(c)
 
    # def _load_mask(self):
    #     mask = np.load('../data/data_clean/mask_clean.npy')
    #     mask_index = torch.tensor(np.where(mask.reshape(-1) == 1)[0])  #np.where()[0]表示行索引，1表示列索引
    #     return mask_index  #返回行索引


def main():
    train_data = load_dataset( mode = 'train' )
    valid_data = load_dataset( mode = 'valid' )
    x, y = train_data.__getitem__( 100 )
    train_data.__len__()
    valid_data.__len__()
    print('x.shape:', x.shape)
    print('y.shape:', y.shape)

    #for ilead in range( x.shape[0] ):
    #    print( ilead, torch.min( x[ ilead ] ), torch.max( x[ilead] ) ) 
    #for ilead in range( y.shape[0] ):
    #    print( ilead, torch.min( y[ ilead ] ), torch.max( y[ilead] ) ) 
    #print(' sst clm == sst truth ? ')
    #print( torch.all( x[-1] == y ) )
    return

if __name__ == '__main__':
    main()
