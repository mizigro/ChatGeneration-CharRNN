import argparse, os, json
import numpy as np
from keras.models import load_model

MODEL_FILE_NAME = 'model.h5'
MODEL_WEIGHT_FILE_NAME = 'weights.h5'
MODEL_C2I_NAME = 'c2i.json' # char to idx
MODEL_I2C_NAME = 'i2c.json' # idx to char


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir',help='Directory of the model')
    parser.add_argument('-s','--samples',help = 'No. of character samples to generate', default = 10000)
    
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise OSError(args.model_dir + ' not found')

    # path variables
    c2i_path = os.path.join(args.model_dir, MODEL_C2I_NAME)
    i2c_path = os.path.join(args.model_dir, MODEL_I2C_NAME)
    model_path = os.path.join(args.model_dir, MODEL_FILE_NAME)

    # check paths are valid
    if not os.path.exists(c2i_path):
        raise FileNotFoundError(c2i_path)
    if not os.path.exists(i2c_path):
        raise FileNotFoundError(i2c_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    # load model and vocabulary
    model = load_model(model_path)
    c2i = json.loads(open(c2i_path,'r').readlines()[0])
    i2c = json.loads(open(i2c_path,'r').readlines()[0])



    # # # TEST PHASE # # #

    # initial input character
    X = np.array([[c2i['h']]])

    # print character
    print( i2c[ str(X[0][0]) ], end = '' )

    for i in range(args.samples):
        Y = model.predict_on_batch(X)

        character_probs = Y[0][0]

        # if new word
        if X[0][0] == c2i[' '] or X[0][0] == c2i['\n']:
            # choose random based on probabilities
            X[0][0] = np.random.choice ( np.arange ( len ( character_probs ) ), 1, p = character_probs ) [0]
        else:
            # choose max probability character
            X[0][0] = np.argmax ( character_probs )
            
        # print character
        print( i2c[ str(X[0][0]) ], end = '' )
