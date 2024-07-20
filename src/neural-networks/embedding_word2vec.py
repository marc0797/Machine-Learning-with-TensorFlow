## Word embedding using word2vec

# This short exercise follows the TensorFlow tutorial 
# on word embeddings using word2vec.

import io
import re
import string

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

class Word2Vec(tf.keras.Model):
    """ Word2Vec subclass model """
    def __init__(self, vocab_size, embedding_dim):
        # Initialize the superclass using overloaded __init__
        super(Word2Vec, self).__init__()
        # Define the embedding layer
        self.target_embedding = layers.Embedding(vocab_size,
            embedding_dim, name="w2v_embedding")
        # Define the context embedding layer
        self.context_embedding = layers.Embedding(vocab_size,
            embedding_dim)

    def call(self, pair):
        target, context = pair
        # Compute the dot product of the target and context embeddings
        target_embedding = self.target_embedding(target)
        context_embedding = self.context_embedding(context)
        # Compute the dot product along the embedding dimensions by defining
        # the tensor product of the target and context embeddings as:
        # target_embedding_{be} * context_embedding_{bce} -> dots_{bc}
        dots = tf.einsum('be,bce->bc', target_embedding, context_embedding)
        return dots
        
def custom_loss(x_logit, y_true):
    """ Custom loss function for word2vec """
    return tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_logit, labels=y_true)

def build_model(vocab_size, embedding_dim):
    """ Build the word2vec model """
    
    # Instantiate the Word2Vec model
    word2vec = Word2Vec(vocab_size, embedding_dim)

    # Compile the model
    word2vec.compile(optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return word2vec

def train_model(word2vec, dataset, epochs):
    """ Train the word2vec model """

    # Define a callback to log training statistics for tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    
    # Train the model
    word2vec.fit(dataset, epochs=epochs, callbacks=[tensorboard_callback])
    return word2vec

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    """ Generate skip-gram pairs with negative sampling for a list of sequences """
    # Initialize lists to hold target words, context words, and labels
    targets, contexts, labels = [], [], []

    # Build the sampling table for vocab_size tokens.
    # This table shows the frequency of each token in the vocabulary.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    for sequence in sequences:
        # Positive skip-gram pairs
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence, 
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0 # we generate the negative samples manually
        )

        # Produce negative skip-grams
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = \
                tf.random.log_uniform_candidate_sampler(
                    true_classes=context_class, # positive class
                    num_true=1, # number of positive context words
                    num_sampled=num_ns, # number of negative context words
                    unique=True, # all the negative samples should be unique
                    range_max=vocab_size, # samples are in [0, vocab_size]
                    seed=seed, # seed for reproducibility
                    name="negative_sampling"
                )
            
            # Create context and label vectors for one target word
            # Each context contains the target word and num_ns negative samples
            # and its corresponding label is 1 for the target word and 0 for the
            # negative samples.
            context = tf.concat([tf.squeeze(context_class,1), 
                        negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")

            # Append to global lists
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels

def preprocess_text(text):
    """ Process the data for vectorization """
    # Lowercase
    lowercase = tf.strings.lower(text)
    # Remove punctuation
    text = tf.strings.regex_replace(lowercase, 
            f"[{re.escape(string.punctuation)}]", "")

    return text

def save_files(weights, vocab):
    """ Save the vectors to files """
    out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()

    # These files can be uploaded to the TensorFlow Projector
    # for visualization.

def main(SEED, AUTOTUNE):
    # Download the text file
    path_to_file = tf.keras.utils.get_file('shakespeare.txt', 
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

    # Read the text file
    with open(path_to_file, 'r') as file:
        text = file.read().splitlines()

    # Print the first few lines
    for line in text[:20]:
        print(line)

    # Use only non-empty lines, for that we filter the empty lines
    # by calculating the length of the string and casting it to a boolean.
    text_ds = tf.data.TextLineDataset(path_to_file).filter(
        lambda x: tf.cast(tf.strings.length(x), bool))

    # Transform to lowercase and remove punctuation
    # print(preprocess_text("Hello, this is a test!"))

    # Define vocabulary size and sequence length
    vocab_size = 4096
    sequence_length = 10

    # Vectorize the data
    text_vectorization = layers.TextVectorization(
        standardize=preprocess_text,
        max_tokens=vocab_size,
        output_sequence_length=sequence_length,
        output_mode='int')
    
    vectorized_text = text_vectorization.adapt(text_ds.batch(1024))

    # Get the inverse vocabulary
    inv_vocab = text_vectorization.get_vocabulary()
    print(inv_vocab[:20])

    # Vectorize the data in the dataset
    text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(text_vectorization).unbatch()

    sequences = list(text_vector_ds.as_numpy_iterator())

    # Generate training data
    targets, contexts, labels = generate_training_data(
        sequences=sequences,
        window_size=2,
        num_ns=4,
        vocab_size=vocab_size,
        seed=SEED)

    targets = np.array(targets)
    contexts = np.array(contexts)
    labels = np.array(labels)

    print(targets.shape, contexts.shape, labels.shape)

    # Efficient batching for the dataset
    BATCH_SIZE = 1024
    BUFFER_SIZE = 10000
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # Define the model
    embedding_dim = 128
    word2vec = build_model(vocab_size, embedding_dim)

    # Train the model
    word2vec = train_model(word2vec, dataset, epochs=20)

    # Embedding analysis
    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    vocab = text_vectorization.get_vocabulary()

    # Save the vectors
    save_files(weights, vocab)

if __name__ == "__main__":
    SEED = 42
    # Prefetch optimization
    AUTOTUNE = tf.data.AUTOTUNE

    main(SEED, AUTOTUNE)