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

    if not os.path.exists(c2i_path):
        raise FileNotFoundError(c2i_path)
    if not os.path.exists(i2c_path):
        raise FileNotFoundError(i2c_path)
    if not os.path.exists(model_path):
        raise FileExistsError(model_path)

    model = load_model(model_path)
    c2i = json.loads(open(c2i_path,'r').readlines()[0])
    i2c = dict(json.loads(open(i2c_path,'r').readlines()[0]))
    
    # N_SAMPLES = args.samples if args.samples!=None else 1000

    X = np.array([[c2i['h']]])

    print( i2c[ str(X[0][0]) ], end = '' )

    for i in range(args.samples):
        Y = model.predict_on_batch(X)

            
        character_probs = Y[0][0]
        if X[0][0] == c2i[' '] or X[0][0] == c2i['\n']:
            X[0][0] = np.random.choice ( np.arange ( len ( character_probs ) ), 1, p = character_probs ) [0]
        else:
            X[0][0] = np.argmax ( character_probs )
            
        print( i2c[ str(X[0][0]) ], end = '' )