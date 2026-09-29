# Lab 1: Word-Level LSTM Text Generator

A small generative language model built with TensorFlow/Keras. It is trained on short children's stories from the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. Given a seed phrase, it continues the
story one word at a time.

```
One day Tom went into the forest and  →  had lots of fun. one day, he saw a big tree ...
```

---

## 1. Project structure

```
Lab1/
├── build_corpus.py        Step 1: downloads TinyStories and writes data/corpus.txt
├── config.py              All hyperparameters and file paths, in one place
├── main.py                Step 3: command-line entry point for text generation
├── requirements.txt       Python dependencies
├── data/
│   └── corpus.txt         Training text (500 stories, one per paragraph)
├── models/
│   ├── text_generator.keras   Trained network weights and architecture
│   └── tokenizer.json         Word ↔ integer id mapping learned from the corpus
└── src/
    ├── preprocessing.py   Text cleaning, tokenizer, training-pair creation
    ├── model.py           The neural network architecture
    ├── train.py           Step 2: training loop (builds, fits and saves the model)
    └── generator.py       Loads the model and generates text (sampling logic)
```

## 2. How the pieces fit together

```
                ┌──────────────────┐
 TinyStories ──►│ build_corpus.py  │──► data/corpus.txt
 (Hugging Face) └──────────────────┘            │
                                                ▼
                ┌──────────────────────────────────────────────┐
                │ src/train.py                                  │
                │   preprocessing.load_corpus()                 │
                │   preprocessing.build_tokenizer()   ──────────┼──► models/tokenizer.json
                │   preprocessing.create_training_sequences()   │
                │   model.build_model()  →  model.fit()  ───────┼──► models/text_generator.keras
                └──────────────────────────────────────────────┘
                                                │
                                                ▼
                ┌──────────────────────────────────────────────┐
 --seed "..."  ►│ main.py  →  src/generator.TextGenerator       │──► generated text
 --temperature  │   normalise seed → predict next word          │
 --top-k        │   → repetition penalty → top-k + temperature  │
 --words        │   → append → repeat                           │
                └──────────────────────────────────────────────┘
```

There are three stages, each with its own script. Each stage reads the previous stage's output from disk, so you
only re-run what you need. For example, you can generate text many times without retraining.

---

## 3. Components in detail

### 3.1 `config.py`: central configuration

Every other module imports its settings from here.

| Setting | Value | Meaning |
|---|---|---|
| `SEQUENCE_LENGTH` | 30 | How many previous words the model sees to predict the next one (its memory window) |
| `VOCAB_SIZE` | 20000 | Maximum number of distinct words the tokenizer keeps |
| `EMBEDDING_DIM` | 192 | Size of the vector that represents each word |
| `LSTM_UNITS_1`, `LSTM_UNITS_2` | 256, 256 | Hidden-state size of the two stacked LSTM layers |
| `DENSE_UNITS` | 256 | Size of the fully connected layer before the output |
| `DROPOUT_RATE` | 0.25 | Fraction of neurons dropped in the dense block during training (regularisation) |
| `BATCH_SIZE` | 64 | Training examples per gradient step |
| `EPOCHS` | 20 | Maximum passes over the data (early stopping may end sooner) |
| `LEARNING_RATE` | 0.0005 | Adam optimiser step size |
| `VALIDATION_SPLIT` | 0.1 | Last 10% of examples held out to measure overfitting |
| `MODEL_PATH`, `TOKENIZER_PATH`, `CORPUS_PATH` | `models/…`, `data/…` | Relative paths, so **scripts must be run from the `Lab1` folder** |

### 3.2 `build_corpus.py`: dataset download

1. Downloads the `train` split of `roneneldan/TinyStories` with the `datasets` library.
2. Takes the first `MAX_STORIES = 500` stories.
3. `clean_text()` removes line breaks and repeated spaces, so each story becomes one line.
4. Writes the stories to `data/corpus.txt`, separated by a blank line.

Increasing `MAX_STORIES` is the single most effective way to get a better model. It needs retraining afterwards.

### 3.3 `src/preprocessing.py`: from raw text to training data

| Function | What it does |
|---|---|
| `load_corpus(path)` | Reads the corpus file; fails clearly if it is missing or empty. |
| `normalize_text(text)` | Lower-cases the text, puts spaces around punctuation (`.,!?;:`) so punctuation becomes its own token, replaces any other symbol with a space and collapses whitespace. The same function is applied to the training corpus **and** to the seed at generation time, so both are tokenised identically. |
| `build_tokenizer(text)` | Fits a Keras `Tokenizer` on the normalised corpus. Each word (and punctuation mark) gets an integer id, with the most frequent words getting the smallest ids. Unknown words map to `<UNK>`. |
| `create_training_sequences(text, tokenizer, sequence_length)` | Converts the corpus into one long list of ids, then slides a window over it. For every position it makes one training pair: the **30 previous tokens** (`x`) and the **next token** (`y`). |
| `save_tokenizer` / `load_tokenizer` | Stores the word↔id mapping as JSON, so generation uses exactly the vocabulary the model was trained with. |

An example of the sliding window, shortened to 4 tokens:

```
tokens:  once upon a time , there was ...
x = [once, upon, a, time]      y = ,
x = [upon, a, time, ,]         y = there
x = [a, time, ,, there]        y = was
```

The model therefore learns a single task: **predict the next word from the previous ones**.

### 3.4 `src/model.py`: the neural network

`build_model()` returns a Keras `Sequential` network:

```
Input (30 token ids)
  → Embedding(vocab, 192)          each id becomes a learned 192-dim vector
  → SpatialDropout1D(0.15)         randomly drops whole embedding channels (regularisation)
  → LSTM(256, return_sequences)    first recurrent layer, outputs a state for every position
  → LayerNormalization
  → LSTM(256)                      second recurrent layer, outputs one summary vector
  → LayerNormalization
  → Dense(256, relu)
  → Dropout(0.25)
  → Dense(vocab, softmax)          a probability for every word in the vocabulary
```

- **Loss:** `sparse_categorical_crossentropy`, meaning "how surprised was the model by the real next word".
- **Optimiser:** Adam with `clipnorm=1.0`. Gradient clipping prevents the exploding gradients that LSTMs are prone to.
- **Metric:** accuracy, meaning how often the single most likely word is the correct one.

### 3.5 `src/train.py`: training loop

1. Loads the corpus, builds the tokenizer and creates `(x, y)` pairs.
2. Builds the model with the sizes from `config.py` and prints a summary.
3. Calls `model.fit()` with three callbacks:
   - **EarlyStopping** (patience 5): stops when validation loss stops improving and restores the best weights.
   - **ReduceLROnPlateau** (factor 0.5, patience 2): halves the learning rate when progress stalls.
   - **ModelCheckpoint:** saves `models/text_generator.keras` whenever validation loss improves.
4. Saves `models/tokenizer.json`.

### 3.6 `src/generator.py`: text generation

The `TextGenerator` class loads the saved model and tokenizer once, then `generate()` runs this loop:

```
1. normalize_text(seed)  → token ids          (unknown words are dropped)
2. repeat until N new words are produced:
   a. take the last 30 ids, left-pad with 0 if shorter
   b. model.predict()   → probability for every word
   c. block padding (id 0) and <UNK>
   d. apply_repetition_penalty()
   e. renormalise so the probabilities sum to 1
   f. sample_top_k()     → choose one word id
   g. append it to the sequence
3. convert ids back to words and format_generated_text()
```

| Method | Purpose |
|---|---|
| `apply_repetition_penalty(probs, tokens, penalty=0.4, window_size=6)` | Multiplies the probability of every word used in the **last 6 tokens** by 0.4, which makes immediate repetition such as "big big big" less likely. Punctuation is exempt. |
| `sample_top_k(probs, temperature, k)` | Keeps only the `k` most likely words, rescales them with the temperature and draws one at random. See the next section. |
| `format_generated_text(tokens)` | Joins tokens with spaces, removes the space before punctuation, adds one after it and capitalises the first letter. |

### 3.7 `main.py`: command-line interface

Parses the arguments, creates a `TextGenerator` and prints the result.

| Argument | Default | Meaning |
|---|---|---|
| `--seed` | (required) | Starting text |
| `--words` | 30 | Number of new words to generate |
| `--temperature` | 0.8 | Sampling temperature (must be > 0) |
| `--top-k` | 8 | Number of candidate words considered at each step |

---

## 4. The sampling parameters explained

At every step the model outputs a probability for every word in the vocabulary. Two parameters control how the next
word is picked from that distribution.

**Top-k: how many candidates?** Only the `k` most likely words are kept.
- `k = 1` always takes the single best word, which is deterministic and quickly repetitive.
- `k = 5` chooses among the 5 best. There is some variety and it rarely goes wrong.
- `k = 20` lets rarer words in, which brings more surprises and more errors.

**Temperature: how evenly are those candidates weighted?** The code takes the log of the kept probabilities, divides
by the temperature and applies softmax again.
- `T < 1` (e.g. 0.3) makes the favourite even more dominant. The text is safe but repetitive.
- `T = 1` uses the model's own probabilities unchanged.
- `T > 1` flattens the distribution. The text is creative but random.

Suppose the candidates are *played* 60%, *ran* 30% and *sky* 10%. At `T = 0.3`, *played* is picked about 90% of the
time. At `T = 1.0` the odds stay 60 / 30 / 10, so *sky* sometimes slips through.

**Words:** the model only sees the last 30 tokens, so longer outputs gradually forget the seed.

### Measured comparison

The seed was *"Once upon a time, there was a little boy named Tom. He went to the forest and"*, with 20 words and
3 runs per setting:

| temperature / top-k | Observation |
|---|---|
| 0.3 / 3 | Same stock phrases every run ("one day, a little girl…") |
| **0.5 / 5** | **Most readable** |
| 0.6 / 8 | Good, slightly looser |
| 0.7 / 10 | More odd phrases ("he saw a big idea") |
| 0.8 / 15 | Grammar breaks, random new names appear |
| 1.0 / 20 | Mostly nonsense |

The recommended setting is `--temperature 0.5 --top-k 5 --words 15`.

---

## 5. How to run

All commands are run **from the `Lab1` folder** with the virtual environment's Python, because the paths in
`config.py` are relative.

```powershell
cd C:\Users\User\PycharmProjects\USM\GenerativeAI\Lab1
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**1. Build the corpus** (only needed once, or after changing `MAX_STORIES`):
```powershell
python build_corpus.py
```

**2. Train the model** (only needed once, or after changing the corpus or `config.py`). Use `-m` so Python can find
`config.py` and the `src` package:
```powershell
python -m src.train
```

**3. Generate text** (as often as you like, with no retraining):
```powershell
python main.py --seed "Once upon a time, there was a little boy named Tom. He went to the forest and" --words 15 --temperature 0.5 --top-k 5
```

To hide TensorFlow's start-up log messages:
```powershell
$env:TF_CPP_MIN_LOG_LEVEL = "3"
```

---

## 6. Limitations and how to improve

| Symptom | Cause | Fix |
|---|---|---|
| Repeated "one day, … he was so excited" | Small model trained on only 500 stories; these are its safest phrases | More stories in `build_corpus.py`, then retrain |
| Grammar errors ("a girl named her friends") | Limited model capacity and data | More data, more epochs, larger LSTM layers |
| Text drifts away from the seed | Memory window is 30 tokens; the LSTM has no long-range planning | Shorter `--words`, or a larger `SEQUENCE_LENGTH` (requires retraining) |
| Seed words ignored | Words not in the vocabulary are dropped silently | Use common story words in the seed |
| Different output every run | Sampling is random and no random seed is fixed | Run several times and pick the best, or set `np.random.seed()` for reproducibility |
| Odd word choices at low top-k | The repetition penalty (0.4, not exposed on the CLI) can push out the fitting word | Use a gentler penalty (0.7–0.9) in `generator.py` |

## 7. Troubleshooting

- **`can't open file '...\main.py'`** means you are in the wrong folder. `cd` into `Lab1` first.
- **`ModuleNotFoundError: tensorflow`** means you are using the system Python instead of `.venv`. Activate the venv, or
  call `.\.venv\Scripts\python.exe` directly.
- **`ModuleNotFoundError: config`** when training means you ran `python src/train.py`. Use `python -m src.train`.
- **`Seed text contains no recognized tokens`** means none of the seed words are in the vocabulary.
- **"TensorFlow GPU support is not available on native Windows"** is only a warning. Training and generation run on the CPU.
