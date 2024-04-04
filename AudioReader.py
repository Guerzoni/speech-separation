import torch
import torchaudio
from utils import handle_scp

def read_pt(fname):
    src = torch.load(fname)
    return src.squeeze()
    
class AudioReader(object):
    '''
        Class that reads .pt format files
        Input as a different scp file address
        Output a matrix of pt files in all scp files.
    '''
    def __init__(self, scp_path):
        super(AudioReader, self).__init__()
        self.index_dict = handle_scp(scp_path)
        self.keys = list(self.index_dict.keys())

    def _load(self, key):
        src = read_pt(self.index_dict[key])
        return src

    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        for key in self.keys:
          yield key, self._load(key)

    def __getitem__(self, index):
        if type(index) not in [int, str]:
            raise IndexError('Unsupported index type: {}'.format(type(index)))
        if type(index) == int:
            num_uttrs = len(self.keys)
            if num_uttrs < index and index < 0:
                raise KeyError('Interger index out of range, {:d} vs {:d}'.format(
                    index, num_uttrs))
            index = self.keys[index]
        if index not in self.index_dict:
            raise KeyError("Missing utterance {}!".format(index))
        return self._load(index)

      
if __name__ == "__main__":
    r = AudioReader('/home/mguerzoni/Conv-TasNet/Conv_TasNet_Pytorch/cv_s2.scp')
    index = 0
    print(r[1].shape)
