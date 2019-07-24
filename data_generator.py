import numpy as np
import tensorflow as tf

class ToySequenceData(object):
    def __init__(self,n_samples=1000,max_seq_len=20,min_seq_len=3,max_value=1000):
        self.n_samples=n_samples
        self.max_seq_len=max_seq_len
        self.min_seq_len=min_seq_len
        self.max_value=max_value
    def generate_random(self):
        sen_len=np.random.randint(self.min_seq_len,self.max_seq_len)
        sen=np.random.randint(0,self.max_value,sen_len)
        return sen
    def generate_linear(self):
        sen_len = np.random.randint(self.min_seq_len, self.max_seq_len)


if __name__=="__main__":
    gen=ToySequenceData()
    for i in range(100):
        print(list(gen.generate()))