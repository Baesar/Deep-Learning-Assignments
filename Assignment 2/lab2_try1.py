#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# Lab 2: Machine Translation with Transformer
# Task 0: Base code (original layers.MultiHeadAttention)
# Task 1: Custom MultiHeadAttention (MIL-style add_weight pattern)
# Task 2: Self-attention visualization (heatmap)
# =============================================================================

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pathlib
import random
import string
import re
import numpy as np

import tensorflow as tf
tf_data = tf.data
tf_strings = tf.strings

import keras
from keras import layers
from keras import ops
from keras.layers import TextVectorization

import matplotlib.pyplot as plt


# =============================================================================
# Data loading and preprocessing
# =============================================================================

text_file = keras.utils.get_file(
    fname="spa-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
    extract=True,
)
text_file = pathlib.Path(text_file).parent / "spa-eng" / "spa.txt"

with open(text_file, encoding="utf-8") as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    eng, spa = line.split("\t")
    eng = "[start] " + eng + " [end]"
    text_pairs.append((spa, eng))

for _ in range(5):
    print(random.choice(text_pairs))

random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")


# =============================================================================
# Vectorization
# =============================================================================

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = 15000
sequence_length = 20
batch_size = 64


def custom_standardization(input_string):
    lowercase = tf_strings.lower(input_string)
    return tf_strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


eng_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)
spa_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
    standardize=custom_standardization,
)
train_eng_texts = [pair[1] for pair in train_pairs]
train_spa_texts = [pair[0] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
spa_vectorization.adapt(train_spa_texts)


# =============================================================================
# Dataset pipeline
# =============================================================================

def format_dataset(eng, spa):
    eng = eng_vectorization(eng)
    spa = spa_vectorization(spa)
    return (
        {
            "encoder_inputs": spa,
            "decoder_inputs": eng[:, :-1],
        },
        eng[:, 1:],
    )


def make_dataset(pairs):
    spa_texts, eng_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf_data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(2048).prefetch(16)


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")


# =============================================================================
# Task 1: Custom MultiHeadAttention layer
#
# Follows the same Keras layer pattern as MILAttentionLayer from:
#   https://keras.io/examples/vision/attention_mil_classification/
#
# MIL pattern used:
#   - All weight matrices created via self.add_weight() in build()
#   - Attention scores computed via matrix ops in call()
#   - Softmax applied to produce weights that sum to 1
#
# The difference: MIL computes a single attention score per instance
#   score = w^T * tanh(V * h)
# Multi-Head Attention computes pairwise scores between all positions:
#   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
# and does this independently for each head, then concatenates.
# =============================================================================

class CustomMultiHeadAttention(layers.Layer):
    """Multi-Head Attention from 'Attention is All You Need' (Vaswani et al., 2017).

    Built using the add_weight() pattern from the Keras MIL attention example.

    Args:
        num_heads: Number of attention heads.
        key_dim: Dimensionality of each head's key/query/value space.

    Call args:
        query: Query tensor of shape (batch, seq_len_q, d_model).
        key: Key tensor of shape (batch, seq_len_k, d_model).
        value: Value tensor of shape (batch, seq_len_v, d_model).
        attention_mask: Optional boolean mask to prevent attention to certain positions.
        return_attention_scores: If True, also return the attention weight matrix.
    """

    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.d_model = num_heads * key_dim

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # --- Weight matrices created via add_weight (MIL pattern) ---

        # Query projection: W_q and b_q
        self.w_q = self.add_weight(
            name="w_q",
            shape=(input_dim, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b_q = self.add_weight(
            name="b_q",
            shape=(self.d_model,),
            initializer="zeros",
            trainable=True,
        )

        # Key projection: W_k and b_k
        self.w_k = self.add_weight(
            name="w_k",
            shape=(input_dim, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b_k = self.add_weight(
            name="b_k",
            shape=(self.d_model,),
            initializer="zeros",
            trainable=True,
        )

        # Value projection: W_v and b_v
        self.w_v = self.add_weight(
            name="w_v",
            shape=(input_dim, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b_v = self.add_weight(
            name="b_v",
            shape=(self.d_model,),
            initializer="zeros",
            trainable=True,
        )

        # Output projection — map back to input dimension, not d_model
        self.w_o = self.add_weight(
            name="w_o",
            shape=(self.d_model, input_dim),  # was (self.d_model, self.d_model)
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b_o = self.add_weight(
            name="b_o",
            shape=(input_dim,),  # was (self.d_model,)
            initializer="zeros",
            trainable=True,
        )
        

    def call(self, query, key, value, attention_mask=None,
             return_attention_scores=False, **kwargs):
        batch_size = ops.shape(query)[0]

        # Step 1: Linear projections for Q, K, V
        # Same idea as MIL's: instance = ops.tensordot(instance, self.v_weight_params, axes=1)
        # but here we project into Q, K, V separately
        Q = ops.matmul(query, self.w_q) + self.b_q  # (batch, seq_q, d_model)
        K = ops.matmul(key, self.w_k) + self.b_k    # (batch, seq_k, d_model)
        V = ops.matmul(value, self.w_v) + self.b_v   # (batch, seq_v, d_model)

        # Step 2: Split into multiple heads
        # (batch, seq, d_model) -> (batch, num_heads, seq, key_dim)
        Q = ops.transpose(
            ops.reshape(Q, (batch_size, -1, self.num_heads, self.key_dim)),
            (0, 2, 1, 3),
        )
        K = ops.transpose(
            ops.reshape(K, (batch_size, -1, self.num_heads, self.key_dim)),
            (0, 2, 1, 3),
        )
        V = ops.transpose(
            ops.reshape(V, (batch_size, -1, self.num_heads, self.key_dim)),
            (0, 2, 1, 3),
        )

        # Step 3: Scaled dot-product attention
        # scores = (Q @ K^T) / sqrt(d_k)
        scale = ops.sqrt(ops.cast(self.key_dim, dtype="float32"))
        scores = ops.matmul(Q, ops.transpose(K, (0, 1, 3, 2))) / scale

        # Apply mask (e.g. padding mask or causal mask)
        if attention_mask is not None:
            if len(ops.shape(attention_mask)) == 3:
                # Expand for head dimension: (batch, seq, seq) -> (batch, 1, seq, seq)
                attention_mask = ops.expand_dims(attention_mask, axis=1)
            # Where mask is False, set score to -inf so softmax gives 0
            scores = ops.where(attention_mask, scores, ops.full_like(scores, -1e9))

        # Step 4: Softmax to get attention weights (sum to 1, just like MIL's alpha)
        # MIL does: alpha = ops.softmax(instances, axis=0)
        # We do:    weights = softmax(scores, axis=-1) — over the key dimension
        attention_weights = ops.softmax(scores, axis=-1)

        # Step 5: Weighted sum of values
        # MIL does: multiply([alpha[i], embeddings[i]])
        # We do:    attention_weights @ V
        context = ops.matmul(attention_weights, V)

        # Step 6: Concatenate heads back together
        # (batch, num_heads, seq, key_dim) -> (batch, seq, d_model)
        context = ops.transpose(context, (0, 2, 1, 3))
        context = ops.reshape(context, (batch_size, -1, self.d_model))

        # Step 7: Final output projection
        output = ops.matmul(context, self.w_o) + self.b_o

        if return_attention_scores:
            return output, attention_weights
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
        })
        return config


# =============================================================================
# Transformer layers (using CustomMultiHeadAttention for Task 1)
# =============================================================================

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        # Task 1: replaced layers.MultiHeadAttention with our custom layer
        self.attention = CustomMultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None, return_attention_scores=False):
        if mask is not None:
            padding_mask = ops.cast(mask[:, None, :], dtype="int32")
        else:
            padding_mask = None

        # Pass return_attention_scores through to our custom layer
        attn_result = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=padding_mask,
            return_attention_scores=return_attention_scores,
        )

        if return_attention_scores:
            attention_output, attention_scores = attn_result
        else:
            attention_output = attn_result

        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        encoder_output = self.layernorm_2(proj_input + proj_output)

        if return_attention_scores:
            return encoder_output, attention_scores
        return encoder_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = ops.shape(inputs)[-1]
        positions = ops.arange(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return ops.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.supports_masking = True

        # Task 1: replaced layers.MultiHeadAttention with our custom layer
        self.attention_1 = CustomMultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim,
        )
        self.attention_2 = CustomMultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim,
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ],
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        decoder_inputs = inputs[0]
        encoder_outputs = inputs[1]

        causal_mask = self.get_causal_attention_mask(decoder_inputs)

        # Masked self-attention (with causal mask)
        attn_output_1 = self.attention_1(
            query=decoder_inputs,
            value=decoder_inputs,
            key=decoder_inputs,
            attention_mask=causal_mask,
        )
        out_1 = self.layernorm_1(decoder_inputs + attn_output_1)

        # Cross-attention (decoder attends to encoder output)
        attn_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
        )
        out_2 = self.layernorm_2(out_1 + attn_output_2)

        # Feed-forward
        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        i = tf.range(seq_len)[:, None]
        j = tf.range(seq_len)[None, :]
        mask = tf.cast(i >= j, dtype=tf.bool)

        mask = tf.expand_dims(mask, axis=0)
        mask = tf.broadcast_to(mask, [batch_size, seq_len, seq_len])
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


# =============================================================================
# Build the model
# =============================================================================
"""
embed_dim = 256
latent_dim = 2048
num_heads = 8
"""
embed_dim = 128    # was 256
latent_dim = 512   # was 2048
num_heads = 4      # was 8

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")

x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)([x, encoded_seq_inputs])
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)

decoder = keras.Model(
    [decoder_inputs, encoded_seq_inputs],
    decoder_outputs,
    name="decoder",
)

final_decoder_outputs = decoder([decoder_inputs, encoder_outputs])

transformer = keras.Model(
    {"encoder_inputs": encoder_inputs, "decoder_inputs": decoder_inputs},
    final_decoder_outputs,
    name="transformer",
)


# =============================================================================
# Train
# =============================================================================

epochs = 1  # Set to at least 30 for convergence

transformer.summary()
transformer.compile(
    "rmsprop",
    loss=keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
    metrics=["accuracy"],
)
transformer.fit(train_ds, epochs=epochs, validation_data=val_ds)


# =============================================================================
# Inference (translation)
# =============================================================================

spa_vocab = eng_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20


def decode_sequence(input_sentence):
    tokenized_input_sentence = spa_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = eng_vectorization([decoded_sentence])[:, :-1]

        predictions = transformer(
            {
                "encoder_inputs": tokenized_input_sentence,
                "decoder_inputs": tokenized_target_sentence,
            }
        )

        sampled_token_index = ops.convert_to_numpy(
            ops.argmax(predictions[0, i, :])
        ).item(0)
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence


test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(30):
    input_sentence = random.choice(test_eng_texts)
    translated = decode_sequence(input_sentence)
    print(f"{input_sentence} = {translated}")


# =============================================================================
# Task 2: Self-attention visualization
#
# We build a separate encoder model that returns attention scores,
# then plot the attention matrix for one head on a Spanish input sentence.
# =============================================================================

def plot_attention_heatmap(sentence, head=0):
    """Plot self-attention heatmap for a given Spanish sentence.

    Args:
        sentence: A Spanish sentence string.
        head: Which attention head to visualize (0 to num_heads-1).
    """
    # Tokenize the sentence
    tokenized = spa_vectorization([sentence])
    token_ids = ops.convert_to_numpy(tokenized[0])

    # Get vocabulary to map IDs back to words
    spa_vocab_list = spa_vectorization.get_vocabulary()

    # Find actual token length (non-zero tokens)
    non_zero = np.where(token_ids == 0)[0]
    seq_len = non_zero[0] if len(non_zero) > 0 else len(token_ids)
    token_labels = [spa_vocab_list[tid] for tid in token_ids[:seq_len]]

    # Run through encoder layers directly in eager mode
    # (no new keras.Model needed — avoids the graph tracing error)
    pos_embed_layer = transformer.get_layer("positional_embedding")
    encoder_layer = transformer.get_layer("transformer_encoder")

    embedded = pos_embed_layer(tokenized)
    _, attention_weights = encoder_layer(embedded, return_attention_scores=True)

    # attention_weights shape: (1, num_heads, seq_len_full, seq_len_full)
    attn_matrix = ops.convert_to_numpy(attention_weights[0, head, :seq_len, :seq_len])

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(attn_matrix, cmap="viridis")

    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(token_labels, rotation=45, ha="right")
    ax.set_yticklabels(token_labels)
    ax.set_xlabel("Key tokens")
    ax.set_ylabel("Query tokens")
    ax.set_title(f"Self-attention heatmap, head {head}")

    plt.colorbar(im, ax=ax, label="Attention weight")
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150)
    plt.show()
    print("Saved attention heatmap to attention_heatmap.png")


# Pick a test sentence and plot its encoder self-attention
test_sentence = random.choice(test_eng_texts)
print(f"\nVisualizing attention for: '{test_sentence}'")
plot_attention_heatmap(test_sentence, head=0)



# =============================================================================
# Task 2: Self-attention visualization
# =============================================================================

def plot_attention_heatmap(sentence, head=0):
    """Plot self-attention heatmap for a given Spanish sentence."""
    # Tokenize the sentence
    tokenized = spa_vectorization([sentence])
    token_ids = ops.convert_to_numpy(tokenized[0])

    # Get vocabulary to map IDs back to words
    spa_vocab_list = spa_vectorization.get_vocabulary()

    # Find actual token length (non-zero tokens)
    non_zero = np.where(token_ids == 0)[0]
    seq_len = non_zero[0] if len(non_zero) > 0 else len(token_ids)
    token_labels = [spa_vocab_list[tid] for tid in token_ids[:seq_len]]

    # Run through encoder layers directly (eager mode, no new Model needed)
    pos_embed_layer = transformer.get_layer("positional_embedding")
    encoder_layer = transformer.get_layer("transformer_encoder")

    embedded = pos_embed_layer(tokenized)
    _, attention_weights = encoder_layer(embedded, return_attention_scores=True)

    # attention_weights shape: (1, num_heads, seq_len_full, seq_len_full)
    attn_matrix = ops.convert_to_numpy(attention_weights[0, head, :seq_len, :seq_len])

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(attn_matrix, cmap="viridis")

    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(token_labels, rotation=45, ha="right")
    ax.set_yticklabels(token_labels)
    ax.set_xlabel("Key tokens")
    ax.set_ylabel("Query tokens")
    ax.set_title(f"Self-attention heatmap, head {head}")

    plt.colorbar(im, ax=ax, label="Attention weight")
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150)
    plt.show()
    print("Saved attention heatmap to attention_heatmap.png")


# Pick a test sentence and plot
test_sentence = random.choice(test_eng_texts)
print(f"\nVisualizing attention for: '{test_sentence}'")
plot_attention_heatmap(test_sentence, head=0)