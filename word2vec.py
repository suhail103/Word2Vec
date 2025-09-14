import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
import unicodedata


# Global Variables
MAX_EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 256

T.random.manual_seed(42);
np.random.seed(42)


class Word2Vec(nn.Module):
    def __init__(self, dictionary_size: int, embedding_dim: int):
        super(Word2Vec, self).__init__()

        self.embedding_layer = nn.Embedding(dictionary_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, dictionary_size)

        # Small Gaussian init
        for layer in [self.embedding_layer, self.output_layer]:
            if not isinstance(layer, nn.Embedding):
                nn.init.normal_(layer.bias, mean=0.0, std=0.05).clamp(-0.1, 0.1)
            nn.init.normal_(layer.weight, mean=0.0, std=0.05).clamp(-0.1, 0.1)
    
    def forward(self, x):
        z = self.embedding_layer(x)
        logits = self.output_layer(z)
        return logits
    
    def inference(self, x):
        embedding_matrix = self.embedding_layer.weight
        word_embedding = embedding_matrix[x]
        return word_embedding


def remove_punctuations(text):
    specific_replacement = [
        ['-', ' '],
    ]
    
    for puntuation, replacement in specific_replacement:
        text = text.replace(puntuation, replacement)

    return "".join(filter(lambda x: not unicodedata.category(x).startswith('P'), text))


def get_text(files):
    text = []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file.readlines():
                no_punctuation_line = remove_punctuations(line)

                for word in no_punctuation_line.split():
                    if len(word) < 2: continue
                    text.append(word.lower())
    return text


def train_model(dataloader, model, criterion, optimizer):
    model.train()
    losses = []

    for batch_idx, (X, y_true) in enumerate(dataloader):
        logits = model(X)
        loss = criterion(logits, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if ((batch_idx + 1) % 100 == 0):
            print(f"Batch: {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f}")

    return np.mean(losses)


def topk_exact_batch(W, k=10, batch=1000):
    W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    V = W.shape[0]
    idx_all = np.empty((V, k), dtype=np.int32)

    for start in range(0, V, batch):
        end = min(start + batch, V)
        sims = np.dot(W_norm[start:end], W_norm.T)
        # Mask self-similarity to avoid picking self as a neighbor
        for i in range(start, end):
            sims[i - start, i] = -np.inf
        idx_all[start:end] = np.argsort(-sims, axis=1)[:, :k]
    return idx_all


def knn_overlap_score(W1, W2, k=10, batch=1000):
    idx1 = topk_exact_batch(W1, k=k, batch=batch)
    idx2 = topk_exact_batch(W2, k=k, batch=batch)
    
    overlap = np.array([
        len(set(idx1[i]) & set(idx2[i])) / k
        for i in range(W1.shape[0])
    ], dtype=np.float32)
    
    return overlap


def detect_stabilization_knn(W_epochs, k=10, threshold=0.9, m=3):
    final_W = W_epochs[-1]
    E = len(W_epochs)
    V = final_W.shape[0]
    stab_epoch = np.full(V, E-1, dtype=int)
    
    # Compare each epoch to final epoch
    scores_per_epoch = []
    for W in W_epochs:
        scores_per_epoch.append(knn_overlap_score(W, final_W, k=k))
    scores_per_epoch = np.stack(scores_per_epoch)  # (E, V)
    
    for w in range(V):
        for e in range(E - m):
            if np.all(scores_per_epoch[e:e+m, w] >= threshold):
                stab_epoch[w] = e
                break
    return stab_epoch, scores_per_epoch


def main():
    files = [f"./Harry_Potter_Books/HP{i}.txt" for i in range(1, 2)]
    text = get_text(files)

    unique_words = []
    word_frequency = {}

    for word in text:
        if word not in unique_words:
            unique_words.append(word)
            word_frequency[word] = 0
        word_frequency[word] += 1

    word_frequency = {word: word_frequency[word] for word in sorted(word_frequency, key=lambda x: word_frequency[x], reverse=True)}
    index_mapping = {word: index for index, word in enumerate(unique_words)}
    word_mapping = {index: word for index, word in enumerate(unique_words)}

    # print(unique_words)
    # print(word_frequency)
    # print(len(word_frequency))

    N = len(text)
    dataset = []
    windows_size = 2

    for i in range(windows_size, N - windows_size):
        # Take the word that before the target word as the context word    
        for j in range(1, windows_size + 1):
            dataset.append([index_mapping[text[i]], index_mapping[text[i - j]]])
            dataset.append([index_mapping[text[i]], index_mapping[text[i + j]]])

    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    # print(f'Using device: {device}')

    dictionary_size = len(unique_words)
    embedding_dims = 128

    model = Word2Vec(dictionary_size, embedding_dims).to(device)

    dataset = T.tensor(dataset, dtype=T.long).to(device=device)
    dataset = TensorDataset(dataset[:, 0], dataset[:, 1])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion =  nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []
    embeddings = []

    for epoch in range(MAX_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        print("[!] Training Model...")
        
        loss = train_model(dataloader, model, criterion, optimizer)
        losses.append(loss)

        print("[+] Model Trained.")

        embeddings.append(model.embedding_layer.weight.detach().cpu().numpy())

        print("-------------------------------\n")

    plt.plot(losses)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("Word2Vec Model Training Loss over Epochs.png")
    plt.show()

    stab_epoch, scores_per_epoch = detect_stabilization_knn(embeddings, k=10, threshold=0.9, m=3)

    bins = np.arange(stab_epoch.min() - 0.5, stab_epoch.max() + 1.5, 1)

    plt.figure(figsize=(8, 4))
    counts, edges, patches = plt.hist(
        stab_epoch,
        bins=bins,
        color='lightcoral',
        edgecolor='black',
        alpha=0.7
    )

    plt.title('Word Stabilization Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Words')
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')

    for count, x in zip(counts, edges[:-1]):
        if count > 0:
            plt.text(x + 0.5, count, int(count), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig("Word Stablilization Epochs Histogram.png")
    plt.show()

    freqs = np.array([word_frequency[word] for word in index_mapping.keys()])
    stab_epochs = np.array([stab_epoch[index_mapping[word]] for word in index_mapping.keys()])
    log_freqs = np.log1p(freqs)

    sorted_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
    top_20 = set(word for word, _ in sorted_words[:10])
    bottom_20 = set(word for word, _ in sorted_words[-10:])

    colors = []
    for word in index_mapping.keys():
        if word in top_20:
            colors.append('red')
        elif word in bottom_20:
            colors.append('blue')
        else:
            colors.append('gray')

    plt.figure(figsize=(10,6))
    plt.scatter(log_freqs, stab_epochs, c=colors, alpha=0.6, label='Other words')

    red_patch = mpatches.Patch(color='red', label='Top 20 Frequent')
    blue_patch = mpatches.Patch(color='blue', label='Bottom 20 Frequent')
    gray_patch = mpatches.Patch(color='gray', label='Other Words')

    plt.legend(handles=[red_patch, blue_patch, gray_patch])
    plt.xlabel('Log(1 + Word Frequency)')
    plt.ylabel('Stabilization Epoch')
    plt.title('Stabilization Epoch vs. Log Word Frequency')
    plt.show()

    rho, p = spearmanr(stab_epoch, np.log1p(freqs))
    print("Spearman rho:", rho, "p-value:", p)


if __name__ == "__main__":
    main()