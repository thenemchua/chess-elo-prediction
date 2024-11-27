import tensorflow as tf
from tensorflow.keras import layers, models

def initialize_model_LSTM (num_filters,position_shape): #maybe add more things

    position_model = models.Sequential # model from position --> in the github Ai-Guess-the-elo The board positions are a (2, seq, 2, 8, 8) tensor.

    ### First Convolution
    position_model.add(layers.Conv2D(filters=num_filters,dilation_rate=(1, 1), kernel_size=(3, 3), padding="same",strides=(1, 1),activation="relu",input_shape=position_shape))

    ### Second Maxpool & Convolution
    position_model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same",strides=(2, 2)))
    position_model.add(layers.Conv2D(dilation_rate=(1, 1), kernel_size=(3, 3), padding="same",strides=(1, 1),activation="relu"))

    ### Third Maxpool
    position_model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same",strides=(2, 2)))

    ### Flattening
    position_model.add(layers.Flatten())

    ## add GEMM layer with tranb = 1 with the evaluation datasource --> PAS SUR J'AI BESOIN D'AIDE
    evaluation_model = models.Sequential # stockfish data (TO CHANGE) in the github Ai-Guess-the-elo The evaluation tensor is a (2, seq, 17) tensor.
    evaluation_model.add(layers.Dense(units=17, activation="relu")) #maybe add --> (input_shape=evaluation_shape))

    ## Concat both model & add layer GEMM with tranb = 1
    ## we use >>The Functional API<< instead of the sequentional API
    combined = layers.Concatenate(axis=1)([position_model.output,evaluation_model.output])
    combined = layers.Dense(units=17)(combined)

    ## Unsqueeze it
    combined = tf.expand_dims(combined, axis=1)

    ## Transpose with perm = 1,0,2
    combined = tf.transpose(combined, perm=[1, 0, 2])

    ## LSTM
    lstm_output, hn, cn = layers.LSTM(64, return_sequences=True)(combined)

    # Squeeze it
    lstm_output = tf.squeeze(lstm_output, axis=1)

    # Final GEMM and Softmax
    final_dense = layers.Dense(units=16, activation="softmax")(lstm_output)

    #  Final Model
    model = models.Model(
        inputs=[position_model.input, evaluation_model.input],
        outputs=final_dense,
    )
    return model


def alain():
    pass
