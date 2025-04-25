# Story Generation using Transformers

## Project Title
Story Generation using Transformers [cite: 1]

## Project Overview
This project centers on the development of a transformer-based model, named "BetterTransformer," designed to generate coherent, engaging, and contextually appropriate children's stories[cite: 4]. The primary goal is to create a versatile model capable of two distinct tasks: unconditional story generation, where narratives are produced without any specific input prompt, and conditional story generation, where stories are generated based on a provided prompt to guide the narrative direction[cite: 5]. The model is trained on a specialized dataset of children's stories sourced from the TinyStories dataset on HuggingFace, ensuring the generated narratives align with the stylistic and thematic characteristics of child-friendly literature[cite: 6]. By leveraging transformer architecture advancements, the project aims to produce narratives that maintain coherence, exhibit creativity, and adapt to varying input conditions, thereby contributing to advancements in natural language processing (NLP) for creative text generation[cite: 7].

## Features
- **Unconditional Story Generation:** Generates stories from scratch, typically starting from a beginning-of-sequence token (<BOS>)[cite: 24]. The model learns to construct full stories in a child-appropriate tone, demonstrating narrative coherence, clear character roles, and a logical progression[cite: 25, 26].
- **Conditional Story Generation:** Generates stories based on a user-provided prompt, such as "Once there was a strong girl named Alyssa"[cite: 19]. The model generates a continuation that stays true to the theme and context of the prompt, maintains consistency in characters, tone, and plot, uses simple vocabulary and short sentences, incorporates relevant themes, and produces a coherent, creative, and natural-sounding story[cite: 20, 21, 22, 23].

## Model Architecture: BetterTransformer
The BetterTransformer is a custom-built Transformer-based language model tailored for autoregressive text generation tasks like story generation[cite: 40]. It is implemented in PyTorch and follows the principles of the original Transformer decoder architecture[cite: 41].

- **Embedding Layer:** Transforms discrete token indices into continuous vector representations using a token embedding layer and a sinusoidal positional encoding module[cite: 42, 43, 44].
- **Transformer Blocks:** A sequence of blocks, each with multi-head self-attention (with causal and padding masks), layer normalization, residual connections, and a position-wise feedforward network[cite: 46, 47, 48, 49, 50].
- **Output Layer:** A linear language modeling head projects hidden states to the vocabulary size[cite: 52].
- **Training Objective:** Uses cross-entropy loss with optional teacher forcing, ignoring padding tokens[cite: 53].
- **Text Generation Strategies:** Supports greedy decoding, temperature sampling, top-k sampling, nucleus (top-p) sampling, and multinomial sampling[cite: 55].

## Dataset: TinyStories
TinyStories is a synthetic dataset of short, simple English-language stories aimed at children[cite: 60].

- **Format & Structure:** Typically 1-5 sentences long with simple, grammatically correct sentences and a logical structure[cite: 62, 63, 64].
- **Language Style:** Uses basic vocabulary, simple grammar, and clear sentence structures, ideal for training smaller models[cite: 65, 66].
- **Diversity:** Covers a wide range of topics (e.g., animals, toys, children, robots)[cite: 67].
- **Size & Tokenization:** Small to medium-sized and efficiently tokenized[cite: 69].
- **Accessibility:** Available through Hugging Face Datasets and similar open-source platforms[cite: 71].

## Methodology

### Data Preprocessing
The TinyStories dataset is loaded and tokenized using the tokenizer from EleutherAI/gpt-neo-1.3B[cite: 78, 80, 81]. Special tokens like <endoftext> (for EOS and BOS) and a custom [PAD] token are handled[cite: 82, 83]. Text sequences are tokenized to a maximum length, EOS tokens are appended, and sequences are padded to a fixed length[cite: 85, 86, 87, 88]. A collation function prepares batches for teacher forcing[cite: 90].

### Training Procedure
The training loop involves configuring CUDA optimizations, enabling Automatic Mixed Precision (AMP) if using a GPU, creating directories for logs and checkpoints, and initializing lists to track losses[cite: 94, 95, 97, 98, 99, 101, 102]. Gradient accumulation is used to simulate larger batch sizes[cite: 104]. The model is evaluated on the validation set before each epoch[cite: 106]. The training loop processes batches with a progress bar, performs forward and backward passes, clips gradients, updates model weights using an optimizer, and zeros out gradients[cite: 110, 111, 112, 113, 116, 117, 120, 121, 122]. CUDA memory is cleared periodically[cite: 123]. Loss and batch timing are logged[cite: 124, 126]. Model checkpoints are saved at the end of each epoch or at specified intervals[cite: 130]. Sample text is generated periodically to assess output quality[cite: 136].

### Evaluation Metrics
The quality of generated stories was evaluated using:
- **Perplexity:** Quantifies the model’s ability to predict the next token[cite: 141].
- **BLEU:** Measures n-gram overlap between generated and reference texts[cite: 143].
- **ROUGE:** Evaluates overlap in n-grams and longest common subsequences, providing scores for ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum[cite: 145].
- **METEOR:** Assesses alignment incorporating synonyms and stemming[cite: 147].

These metrics were calculated during validation, and results were saved to a CSV file[cite: 149, 150].

### Experimental Setup
The experimental setup involved orchestrating training runs with the `train_model` function[cite: 152]. The BetterTransformer model was configured with specific numbers of layers (4), attention heads (4), embedding dimension (256), vocabulary size (50,258), and sequence length (384)[cite: 153, 154, 155, 156]. The dataset used was a collection of approximately 8,000 children’s stories from Project Gutenberg, tokenized using the AutoTokenizer based on GPT-Neo 1.3B[cite: 157]. Data preprocessing included handling special tokens and filtering malformed stories[cite: 158]. Data splitting varied by experiment, with validation data fixed at 500 stories[cite: 159]. Training configurations included batch size (32), learning rate (0.0001), and epochs (default 10)[cite: 160, 161]. Checkpointing and generation of text samples were enabled[cite: 162, 163]. Evaluation metrics were computed and logged per epoch[cite: 165, 166].

## Experiments and Results
Several experiments were conducted to evaluate different model architectures and the impact of dataset size.

- **Experiment 1:** Initial exploration with small Transformer models on a subset of the TinyStories dataset (25%)[cite: 172, 173]. Model 3 (4 layers, 4 heads, 256 embedding size) offered the best trade-off between story quality and computational cost[cite: 231].
- **Experiment 2:** Retraining smaller Transformer variants on a smaller dataset subset (10%) with a reduced learning rate (0.0005) and proper perplexity calculation[cite: 237, 238]. Model 2 (6 layers, 4 heads) demonstrated the optimal trade-off between representational capacity and overfitting resistance[cite: 279]. Different decoding strategies were also evaluated, with Top-k sampling improving lexical diversity and plot novelty[cite: 276].
- **Experiment 3:** Scaling up training with the best architecture (Model 1 from Exp 1, 4 layers, 4 attention heads, 256 embedding size) to 40% of the TinyStories dataset[cite: 286, 287, 290]. Dataset scaling significantly lowered perplexity and boosted generation metrics[cite: 306].
- **Experiment 4:** Final training using the largest subset (70% of TinyStories) with revised generation code for strict prompt adherence[cite: 312]. Scaling to 70% further reduced perplexity and improved all generation metrics[cite: 319]. The generation code fix fully resolved conditional prompt adherence issues[cite: 320].

## References
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30. [cite: 326]
- Hugging Face. (n.d.). Transformers: State-of-the-art Natural Language Processing. Retrieved from [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers) [cite: 327]
- Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32. [cite: 328]
- Project Gutenberg. (n.d.). Children's Stories Collection. Retrieved from [https://www.gutenberg.org](https://www.gutenberg.org) [cite: 329]
- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 38-45. [cite: 331]
