from keras.layers import LSTM, Dense, Input, Embedding, Dropout, TimeDistributed
from keras.optimizers import RMSprop, Adam
from keras.models import Model, load_model


def get_char_rnn_model(batch_size, seq_len, lstm_cell_size, vocab_size):

    inputs = Input ( batch_shape = ( batch_size, seq_len ) )

    embed = Embedding ( vocab_size, vocab_size, input_length = seq_len )
    lstm_1_in = embed ( inputs )

    # LSTM Units
    lstm_1_out = LSTM ( lstm_cell_size, return_sequences = True, stateful = True ) ( lstm_1_in )
    lstm_1_out = Dropout ( 0.2 ) ( lstm_1_out )
    lstm_2_out = LSTM ( lstm_cell_size, return_sequences = True, stateful = True ) ( lstm_1_out )
    lstm_2_out = Dropout ( 0.2 ) ( lstm_2_out )
    lstm_3_out = LSTM ( lstm_cell_size, return_sequences = True, stateful = True ) ( lstm_2_out )
    lstm_3_out = Dropout ( 0.2 ) ( lstm_3_out )

    # Final Dense Layer
    outputs = TimeDistributed ( Dense ( vocab_size, activation = 'softmax' ) ) ( lstm_3_out )
    model = Model ( inputs, outputs )

    # Model Generation
    op = RMSprop ( lr = 0.001, rho = 0.9, epsilon = None, decay = 0.0 )
    model.compile ( optimizer = op, loss = 'categorical_crossentropy', metrics = ['accuracy'] )

    return model


def get_final_model(lstmSize, vocab_size):
    return get_char_rnn_model(1, 1, lstmSize, vocab_size)