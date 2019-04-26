import argparse, os, sys, json
import models
import vocabs

MODEL_FILE_NAME = 'model.h5'
MODEL_WEIGHT_FILE_NAME = 'weights.h5'
MODEL_C2I_NAME = 'c2i.json' # char to idx
MODEL_I2C_NAME = 'i2c.json' # idx to char

# save object in json format
def save_json(obj, path):
    open(path,'w').write(json.dumps(obj))

def train(txt, model_dir, replace):
    lstm_cell_size = 256
    steps_per_epoch = 200
    seq_len = 100
    batch_size = 32
    epochs = 10

    # reject text if too small
    if len(txt) < steps_per_epoch * (seq_len + 1):
        raise AssertionError('Too few texts in file :'+str(len(txt))+' Decrease steps_per_epoch : '+str(steps_per_epoch)+' and seq_len : '+str(seq_len))

    # path variables
    c2i_path = os.path.join(model_dir, MODEL_C2I_NAME)
    i2c_path = os.path.join(model_dir, MODEL_I2C_NAME)
    model_wt_path = os.path.join(model_dir, MODEL_WEIGHT_FILE_NAME)
    model_path = os.path.join(model_dir, MODEL_FILE_NAME)

    # vocabulary
    vocab = vocabs.Vocab(txt)
    vocab_size = len(vocab.c2i)

    save_json(vocab.c2i, c2i_path) 
    save_json(vocab.i2c, i2c_path) 

    # get model architectures
    model = models.get_char_rnn_model(batch_size, seq_len, lstm_cell_size, vocab_size)
    final_model = models.get_final_model(lstm_cell_size, vocab_size)

    if not replace:
        # Try reusing saved weights
        try:
            model.load_weights(model_wt_path)
            print('\nModel already exists. Reusing weights.')
        except OSError:
            print('\nNo existing models found. Training new model ...')
    
    # text data generator
    data_generator = vocab.new_data_generator(batch_size, steps_per_epoch, seq_len)

    # final training phase
    for epoch in range(epochs):
        model.reset_states()
        print('Epoch: ', epoch+1)
        model.fit_generator(data_generator, steps_per_epoch = steps_per_epoch, epochs = 1)

        print('Saving Model ...',end='\r')
        model.save_weights(model_wt_path)
        final_model.load_weights(model_wt_path)
        final_model.save(model_path)
        print('Model Saved !\t')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text_file_path',help='Path to text file')
    parser.add_argument('-m','--model_name',type=str, help='Specify the name for the model saved')
    parser.add_argument('-r','--replace',action='store_true', help='To replace existing model data')

    args = parser.parse_args()

    # text file
    if not os.path.exists(args.text_file_path):
        raise FileNotFoundError(args.text_file_path)

    # model path
    model_name = args.model_name if args.model_name != None else 'model' 

    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_dir = os.path.join(model_dir, model_name) 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # get text data from file
    txt = ''.join(open(args.text_file_path,'r',errors='ignore').readlines())

    # training phase
    train(txt, model_dir, replace = args.replace)