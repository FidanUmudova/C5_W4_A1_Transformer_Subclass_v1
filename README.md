# Transformer Architecture from Scratch (TensorFlow / Keras)

A modular, step-by-step implementation of the classic **Transformer** architecture based on the seminal paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) by Vaswani et al., built using **TensorFlow** and **Keras**.

---

##  Project Architecture & Components

This project implements all core components of the Transformer model from the ground up:

1. **Positional Encoding (`get_angles` & `positional_encoding`)**
   - Generates sine and cosine positional encodings to inject sequence order information into token embeddings.
   - Computes frequencies using dimension spans and applies alternating trigonometric functions to even and odd indices.

2. **Scaled Dot-Product Attention (`scaled_dot_product_attention`)**
   - Calculates attention weights via matrix multiplication of Queries ($Q$) and Keys ($K$).
   - Scales scores by $\sqrt{d_k}$ to stabilize gradients.
   - Supports optional padding and look-ahead masking by setting masked regions to $-10^9$ prior to softmax normalization.
   - Computes final context representations by multiplying weights with Values ($V$).

3. **Encoder Stack (`EncoderLayer` & `Encoder`)**
   - **EncoderLayer**: Combines multi-head self-attention with residual connections, layer normalization, and a position-wise feed-forward network with dropout.
   - **Encoder**: Embeds input tokens, scales them by $\sqrt{d_{\text{model}}}$, adds positional encodings, applies dropout, and passes the tensor sequentially through a stack of $N$ Encoder Layers.

4. **Decoder Stack (`DecoderLayer` & `Decoder`)**
   - **DecoderLayer**: Composed of three blocks:
     1. Masked multi-head self-attention with look-ahead masks.
     2. Encoder-decoder multi-head attention (taking $Q$ from Block 1 and $K, V$ from the Encoder output).
     3. Position-wise feed-forward network with residual connections and layer normalization.
   - **Decoder**: Embeds target tokens, adds positional encodings, applies dropout, and passes through a stack of $N$ Decoder Layers while keeping track of attention weights.

5. **Full Transformer Model (`Transformer`)**
   - Integrates both the `Encoder` and `Decoder` into an end-to-end Keras model.
   - Concludes with a final linear (`Dense`) layer and softmax activation to output token probability distributions over the target vocabulary.

---

