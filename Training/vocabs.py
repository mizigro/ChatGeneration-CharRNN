import random
import numpy as np

class Vocab:
    def __init__(self, txt):
        self.txt = txt
        self.c2i, self.i2c = self.get_dicts()

    def get_dicts(self):
        c2i = dict()
        i2c = dict()

        for char in self.txt:
            if char not in c2i:
                i2c [len(c2i)] = char
                c2i [ char ] = len(c2i)
                
        return c2i, i2c


    # def save_dict(_dict, path):
    #     open(path,'w').write(json.dumps(_dict))

    def C2I ( self, txt ):
        return [ self.c2i [char] for char in txt ]

    def I2CEye ( self, ii ):
        return self.I2C ( self.unEye (  ii ) )

    def unEye ( self, ii ):
        return [ np.argmax ( i ) for i in ii ] 

    def I2C ( self, ii ):
        return ''.join ( [ self.i2c [ i ] for i in ii ] )

    def new_data_generator(self, batch_size, steps_per_epoch, seq_len):
        while True:
            batchCursor = [ random.randint ( 0, len ( self.txt ) - steps_per_epoch * ( seq_len + 1 ) ) for i in range ( batch_size ) ] 

            for _ in range ( steps_per_epoch ) : 
                X, Y = [], []

                for batchId in range(batch_size):
                    i = batchCursor [ batchId ] 

                    # idx for X
                    ii = self.C2I( self.txt [ i : i + seq_len ] )
                    X.append( ii )

                    # one hot for Y
                    ii = self.C2I ( self.txt [ i + 1 : i + seq_len + 1 ] )
                    Y.append( np.eye ( len ( self.c2i ) ) [ ii ] )

                    batchCursor [ batchId ] += seq_len
                    
                yield np.array(X), np.array(Y)
