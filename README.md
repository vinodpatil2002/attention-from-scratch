# Attention from Scratch

This repository demonstrates a minimal implementation of attention mechanisms in a Transformer-like architecture. Below is an overview of each component in the model.py file.

## Classes Overview

1. **InputEmbeddings**  
   • Creates token embeddings using PyTorch's nn.Embedding.  
   • Scales them by √(d_model) for better gradient flow.

2. **PositionalEncoding**  
   • Adds positional information to embeddings using sin/cos functions.  
   • Registered as a buffer to remain unchanged during backprop.

3. **LayerNormalisation**  
   • Applies layer normalization to stabilize training.  
   • Uses learned α (scale) and β (bias) parameters.

4. **FeedForwardBlock**  
   • Projects input to higher-dimensional space, applies ReLU, then projects back.  
   • Uses dropout to prevent overfitting.

5. **MultiHeadAttentionBlock**  
   • Splits embeddings into multiple heads, calculates scaled dot-product attention, and projects outputs.  
   • Applies dropout to attention weights.

6. **ResidualConnectionBlock**  
   • Adds a skip connection plus dropout.  
   • Normalizes input before passing to the sublayer.

7. **EncoderBlock**  
   • Combines self-attention and feed-forward layers.  
   • Uses residual connections to stabilize training.  

8. **Encoder**  
   • Stacks multiple EncoderBlocks and applies a final normalization.  
   • Iteratively refines the input representation using attention and feed-forward blocks.

9. **DecoderBlock**  
   • Handles self-attention on the target sequence and cross-attention with the encoder output.  
   • Uses three residual connections to integrate sublayers.

10. **Decoder**  
   • Stacks multiple DecoderBlocks.  
   • Applies a final normalization step before output.

11. **ProjectionLayer**  
   • Projects the final output to vocab_size dimensions.  
   • Applies log_softmax for probability distribution.

## Usage

• Import and instantiate these classes as building blocks for your Transformer.  
• Adjust hyperparameters such as d_model, seq_len, vocab_size, dropout, and number of heads as needed.  
• Make sure to install PyTorch and have the correct environment setup.