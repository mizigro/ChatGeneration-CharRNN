# Messenger Chat Generation (Char-RNN Model)

Char-RNN model run on dialogue between <a href="https://github.com/mizimo/ChatGeneration-CharRNN/raw/master/Training/data/conversation.txt">Socrates and Euthyphro (Plato's "Euthyphro") </a>


## Sample Text Conversation

<img src="https://github.com/mizimo/ChatMessenger-CharRNN/raw/master/Server/sample.png" width="100%">

## Running the Server

Install `Flask (Python)`. Go to the `Server`folder and run `export FLASK_APP=app.py` and `flask run` on the terminal, open the link in a browser and enjoy!

## Training 

For training on a different conversation, first collect a conversation file similar to the format given in `conversation.txt` file in the data folder. 

Run `python3 train.py path/to/text/file --model_name model/name` using `train.py` in the Training folder. This will generate a model folder with the given model name (default name : 'model') after the training process is complete. It contains the `c2i.json`, `i2c.json`, `model.h5` and `weights.h5` files of the model.

Test the model using `python3 test.py model/folder/path --samples 3000`. The samples argument specifies the number of characters to predict.

## Moving trained model to Server

Generate TensorflowJS files using `tensorflowjs_converter --input_format keras path/to/model.h5 target/dir/path` to convert the model for the server app *(TensorflowJS may required use of virtualenv due to version conflicts in the dependencies)*.

Copy the `model.json` file generated, alongwith the `c2i.json` and `i2c.json` files of the model, to the static folder of the server.

Specify the path to those files in the `model.options.json` file found inside the `static` folder. Also, specify all the users' names as used in the conversation text. You can even modify the number of character samples to be generated as well as the initial character used for prediction.

Run the sever as described before.
