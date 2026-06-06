import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dropout, Dense
from tensorflow.keras.models import Model
from src.utils.config import Config

def build_semantic_level_cnn(sequence_length=Config.MAX_TOKENS, embedding_dim=Config.EMBEDDING_DIM, filters=Config.CNN_FILTERS, kernel_size=Config.KERNEL_SIZE, dropout_rate=Config.DROPOUT_RATE):
    """
    Constructs the Semantic-level DH-CNN branch
    
    Args:
        sequence_length (int): The number of nodes kept per file Fixed at 50
        embedding_dim (int): The dimension of the Node2vec vectors 50
        filters (int): The number of filters for the convolution.
        kernel_size (int): The size of the convolution window.
        dropout_rate (float): The rate of neuron dropout to avoid overfitting.
        
    Returns:
        tf.keras.Model: model non compiled representing semantic branch
    """
    
    # Input Layer
    inputs = Input(shape=(sequence_length, embedding_dim), name="semantic_graph_input")
    
    # Conv1D Layer
    x = Conv1D(
        filters=filters, 
        kernel_size=kernel_size, 
        activation='relu', 
        name="semantic_conv1d"
    )(inputs)
    
    # Global Max-Pooling Layer
    x = GlobalMaxPooling1D(name="semantic_maxpool")(x)
    
    # Dropout Layer
    x = Dropout(dropout_rate, name="semantic_dropout")(x)
    
    # Fully Connected Layer - fusion
    outputs = Dense(Config.FC_UNITS, activation='sigmoid', name="semantic_fully_connected")(x)
    
    # model creation
    model = Model(inputs=inputs, outputs=outputs, name="Semantic_Level_DH_CNN")
    
    return model