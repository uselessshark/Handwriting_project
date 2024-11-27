
#Импорт необходимых библиотек
import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter
import matplotlib.pyplot as plt
from jiwer import wer, cer
import warnings

warnings.filterwarnings("ignore")



"""
 Parses the words.txt file to extract word IDs, labels, and the full line information.

 Args:
     words_file_path (str): Path to the words.txt file.
     include_err (bool): Whether to include words labeled as 'err'.

 Returns:
     list: A list of tuples containing (word_id, word_label, full_line).
 """

def parse_words_file(words_file_path, include_err=False):

    samples = []
    with open(words_file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue  # Skip comments
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            parts = line.split(' ')
            if len(parts) >= 9:
                word_id = parts[0]
                word_label = parts[-1]
                status = parts[1]
                if status == 'err' and not include_err:
                    continue  # Skip 'err' labeled words unless included
                samples.append((word_id, word_label, line))
    return samples


def get_image_path(base_path, word_id):
    """
    Constructs the image path from the base path and word ID.

    Args:
        base_path (str): Base directory containing the word images.
        word_id (str): Unique identifier for the word.

    Returns:
        str: Full path to the image file.
    """
    parts = word_id.split('-')
    subdir1 = parts[0]
    subdir2 = '-'.join(parts[:2])
    image_name = f"{word_id}.png"
    image_path = os.path.join(base_path, subdir1, subdir2, image_name)
    return image_path

# Paths to your dataset
words_file_path = 'words.txt'  # Update with your actual path if different
base_image_path = 'words'  # Update with your actual path if different

# Parse the words.txt file, excluding 'err' labeled words
samples = parse_words_file(words_file_path, include_err=False)

# Image dimensions
image_width = 160
image_height = 40

# Data augmentation transforms for training
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(degrees=10, fill=(0,)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Corrected placement
    transforms.Normalize((0.5,), (0.5,))
])

# Transforms for validation (no random augmentations)
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Build character vocabulary
chars = set()

# Lists to hold images and labels
images = []
labels = []
word_ids = []
full_lines = []

# Load and preprocess images
for word_id, word_label, full_line in samples:
    image_path = get_image_path(base_image_path, word_id)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue
    try:
        # Load the image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((image_width, image_height), Image.LANCZOS)
        img = np.array(img)
        # Note: Augmentations are applied in the DataLoader via transform
        images.append(img)
        labels.append(word_label)
        word_ids.append(word_id)
        full_lines.append(full_line)
        chars.update(list(word_label))
    except (IOError, SyntaxError, UnidentifiedImageError) as e:
        print(f"Skipping corrupted image: {image_path}. Error: {e}")
        continue

# Convert images and labels to lists
X = images
y = labels

# Build character mappings
chars = sorted(list(chars))
char_to_num = {c: i + 1 for i, c in enumerate(chars)}  # Start indices from 1
num_to_char = {i + 1: c for i, c in enumerate(chars)}
num_to_char[0] = ''  # For blank character in CTC

# Encode labels
encoded_labels = []
label_lengths = []
for label in y:
    label_seq = [char_to_num[c] for c in label]
    encoded_labels.append(label_seq)
    label_lengths.append(len(label_seq))

# Ensure no label contains the blank index (0)
for idx, lbl in enumerate(encoded_labels):
    if 0 in lbl:
        print(f"Label contains blank index at sample {idx}")

# Check class distribution
label_counter = Counter([char for label in y for char in label])
print("Class Distribution:", label_counter)

# Split data into training and validation sets without stratification to avoid the earlier error
X_train, X_val, y_train_enc, y_val_enc, y_train_str, y_val_str, train_label_lengths, val_label_lengths, word_train_ids, word_val_ids, word_train_lines, word_val_lines = train_test_split(
    X, encoded_labels, y, label_lengths, word_ids, full_lines, test_size=0.2, random_state=42, shuffle=True)


# ----------------------------
# 2. Custom Dataset and DataLoader
# ----------------------------

class HandwritingDataset(Dataset):
    """
    Custom Dataset for Handwriting Recognition.

    Args:
        images (list): List of preprocessed images.
        labels (list): List of encoded labels.
        label_lengths (list): List of label lengths.
        word_ids (list): List of word IDs.
        full_lines (list): List of full lines from words.txt.
    """

    def __init__(self, images, labels, label_lengths, word_ids, full_lines, transform=None):
        self.images = images
        self.labels = labels
        self.label_lengths = label_lengths
        self.word_ids = word_ids
        self.full_lines = full_lines
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # [H, W]
        label = self.labels[idx]
        label_length = self.label_lengths[idx]
        word_id = self.word_ids[idx]
        full_line = self.full_lines[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        return image, label, label_length, word_id, full_line


def collate_fn(batch):
    """
    Custom collate function to handle batches with variable-length labels.

    Args:
        batch (list): List of tuples returned by the Dataset's __getitem__.

    Returns:
        tuple: Batched images, labels, input lengths, label lengths, word IDs, and full lines.
    """
    images, labels, label_lengths, word_ids, full_lines = zip(*batch)

    # Stack images
    images = torch.stack(images)  # [batch, 1, H, W]

    # Concatenate labels
    labels_concat = []
    for lbl in labels:
        labels_concat.extend(lbl)
    labels_tensor = torch.tensor(labels_concat, dtype=torch.long)

    # Label lengths
    label_lengths_tensor = torch.tensor(label_lengths, dtype=torch.long)

    # Input lengths (based on CNN architecture)
    # Assuming output width after CNN is 34 for image_width=160 and image_height=40
    input_lengths = torch.full(size=(len(images),), fill_value=34, dtype=torch.long)

    return images, labels_tensor, input_lengths, label_lengths_tensor, word_ids, full_lines


batch_size = 64

train_dataset = HandwritingDataset(X_train, y_train_enc, train_label_lengths, word_train_ids, word_train_lines,
                                   transform=train_transforms)
val_dataset = HandwritingDataset(X_val, y_val_enc, val_label_lengths, word_val_ids, word_val_lines,
                                 transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4,
                          pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4,
                        pin_memory=True)


# ----------------------------
# 3. Model Definition
# ----------------------------

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for Handwriting Recognition.

    Args:
        imgH (int): Height of the input image.
        nc (int): Number of input channels.
        nclass (int): Number of output classes (including CTC blank).
        nh (int): Number of hidden units in LSTM layers.
    """

    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),  # [batch, 64, H, W]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [batch, 64, H/2, W/2]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [batch, 128, H/2, W/2]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [batch, 128, H/4, W/4]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [batch, 256, H/4, W/4]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), padding=(0, 1)),  # [batch, 256, H/8, W/4 +1]

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # [batch, 256, H/8, W/4 +1]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), padding=(0, 1)),  # [batch, 256, H/16, W/4 +2]
        )

        # Adjusted Linear layer input size from 256*(H//16)=256*2=512 to nh=1024
        self.rnn_linear = nn.Linear(256 * (imgH // 16), nh)
        self.rnn_relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.lstm1 = nn.LSTM(nh, nh, bidirectional=True)
        self.lstm2 = nn.LSTM(nh * 2, nh, bidirectional=True)
        self.fc = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, 1, H, W].

        Returns:
            torch.Tensor: Log probabilities of shape [W', batch, nclass].
        """
        conv = self.cnn(x)  # [batch, 256, H/16, W']

        # Get the dimensions
        b, c, h, w = conv.size()
        # Permute to [W', batch, C, H']
        conv = conv.permute(3, 0, 1, 2)  # [W', batch, 256, H/16]

        # Flatten [256, H/16] into [256 * H/16]
        conv = conv.contiguous().view(w, b, -1)  # [W', batch, 256 * H/16]

        # Pass through Linear layer and ReLU
        conv = self.rnn_linear(conv)  # [W', batch, nh]
        conv = self.rnn_relu(conv)
        conv = self.dropout(conv)

        # Pass through LSTM layers
        output, _ = self.lstm1(conv)  # [W', batch, nh*2]
        output, _ = self.lstm2(output)  # [W', batch, nh*2]

        # Pass through Fully Connected layer
        output = self.fc(output)  # [W', batch, nclass]
        return output.log_softmax(2)  # Apply log_softmax on class dimension


# Initialize model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

imgH = image_height
nc = 1  # Number of input channels
nclass = len(chars) + 1  # Number of classes (including CTC blank)
nh = 1024  # Increased hidden size for better capacity

model = CRNN(imgH, nc, nclass, nh).to(device)


# Initialize weights
def init_weights(m):
    """
    Initializes weights of convolutional and linear layers using Xavier uniform initialization.

    Args:
        m (nn.Module): Layer to initialize.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


model.apply(init_weights)

# ----------------------------
# 4. Training Setup
# ----------------------------

# Define loss and optimizer
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Gradient clipping parameter
max_grad_norm = 5

# Initialize training parameters for Early Stopping
num_epochs = 200  # Increased number of epochs
patience = 15  # Increased patience to allow more epochs for improvement
best_val_loss = float('inf')  # Initialize best validation loss
trigger_times = 0  # Counter for early stopping

# Lists to store loss and CER for plotting
train_losses = []
val_losses = []
val_cers = []  # To store validation CER per epoch


# ----------------------------
# 5. Training Loop
# ----------------------------

def decode_predictions(outputs):
    """
    Performs greedy decoding on the model outputs.

    Args:
        outputs (torch.Tensor): Log probabilities of shape [W', batch, nclass].

    Returns:
        list: List of decoded sequences for each sample in the batch.
    """
    decoded_sequences = []
    outputs = outputs.transpose(0, 1)  # [batch, W', nclass]
    for output in outputs:
        probs = output.detach().cpu()
        max_indices = probs.argmax(dim=1)
        max_indices = max_indices.numpy()
        # Remove consecutive duplicates and blanks (0)
        decoded = []
        prev_idx = -1
        for idx in max_indices:
            if idx != prev_idx and idx != 0:
                decoded.append(idx)
            prev_idx = idx
        decoded_sequences.append(decoded)
    return decoded_sequences


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels, input_lengths, label_lengths, word_ids_batch, full_lines_batch) in enumerate(
            train_loader):
        images = images.to(device)
        labels = labels.to(device)
        input_lengths = input_lengths.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # [W', batch, nclass]

        # Ensure log_probs are in [W', batch, nclass]
        log_probs = outputs  # [W', batch, nclass]

        # Compute loss
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        running_loss += loss.item()

        # Print loss every 100 steps
        if (batch_idx + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        total_cer = 0.0
        total_samples = 0
        misrecognized_words = []  # List to store details of misrecognized words
        for images, labels, input_lengths, label_lengths, word_ids_val, full_lines_val in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            outputs = model(images)  # [W', batch, nclass]
            log_probs = outputs  # [W', batch, nclass]

            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            val_loss += loss.item()

            # Decoding using Greedy Decoding
            decoded_sequences = decode_predictions(log_probs)

            for i in range(len(decoded_sequences)):
                pred_label = ''.join([num_to_char.get(idx, '') for idx in decoded_sequences[i]])
                true_label = y_val_str[i]
                sample_wer = wer(true_label, pred_label)
                sample_cer = cer(true_label, pred_label)  # Compute CER
                total_cer += sample_cer
                total_samples += 1

                if sample_wer > 0:
                    misrecognized_words.append({
                        'word_id': word_val_ids[i],
                        'true_label': true_label,
                        'predicted_label': pred_label,
                        'full_line': word_val_lines[i],
                        'wer': sample_wer,
                        'cer': sample_cer
                    })

    avg_val_loss = val_loss / len(val_loader)
    avg_cer = total_cer / total_samples
    val_losses.append(avg_val_loss)
    val_cers.append(avg_cer)  # Store CER
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation CER: {avg_cer:.4f}")

    # Step the scheduler
    scheduler.step(avg_val_loss)

    # Early Stopping Check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_crnn_model.pth')
        print("Model saved!")
        trigger_times = 0  # Reset trigger_times because validation loss improved
    else:
        trigger_times += 1
        print(f"Trigger times: {trigger_times}")
        if trigger_times >= patience:
            print("Early stopping!")
            break

    # Save misrecognized words after each epoch
    with open(f'misrecognized_words_epoch_{epoch + 1}.txt', 'w') as f:
        for word in misrecognized_words:
            f.write(f"Word ID: {word['word_id']}\n")
            f.write(f"Full Line: {word['full_line']}\n")
            f.write(f"True Label: {word['true_label']}\n")
            f.write(f"Predicted Label: {word['predicted_label']}\n")
            f.write(f"Word Error Rate: {word['wer']}\n")
            f.write(f"Character Error Rate: {word['cer']}\n")
            f.write('---\n')

# ----------------------------
# 6. Visualization
# ----------------------------

# Plot training and validation loss along with validation CER
plt.figure(figsize=(18, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()

# Plot Validation CER
plt.subplot(1, 2, 2)
plt.plot(range(1, len(val_cers) + 1), val_cers, label='Validation CER', color='green')
plt.xlabel('Epochs')
plt.ylabel('Character Error Rate (CER)')
plt.title('Validation CER Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# ----------------------------
# 7. Evaluation on Validation Set
# ----------------------------

model.eval()
with torch.no_grad():
    total_wer = 0.0
    total_cer = 0.0
    total_samples = 0
    misrecognized_words = []
    for images, labels, input_lengths, label_lengths, word_ids_val, full_lines_val in val_loader:
        images = images.to(device)
        outputs = model(images)  # [W', batch, nclass]
        log_probs = outputs  # [W', batch, nclass]

        # Decoding using Greedy Decoding
        decoded_sequences = decode_predictions(log_probs)

        for i in range(len(decoded_sequences)):
            pred_label = ''.join([num_to_char.get(idx, '') for idx in decoded_sequences[i]])
            true_label = y_val_str[i]
            sample_wer = wer(true_label, pred_label)
            sample_cer = cer(true_label, pred_label)  # Compute CER
            total_wer += sample_wer
            total_cer += sample_cer
            total_samples += 1

            if sample_wer > 0:
                misrecognized_words.append({
                    'word_id': word_val_ids[i],
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'full_line': word_val_lines[i],
                    'wer': sample_wer,
                    'cer': sample_cer
                })

            print(f"True label: {true_label}")
            print(f"Predicted: {pred_label}")
            print(f"Word Error Rate: {sample_wer}")
            print(f"Character Error Rate: {sample_cer}")
            print('---')
        break  # Only display the first batch

    average_wer = total_wer / total_samples
    average_cer = total_cer / total_samples
    print(f"Average Word Error Rate on Validation Set: {average_wer:.4f}")
    print(f"Average Character Error Rate on Validation Set: {average_cer:.4f}")

    # Save misrecognized words to a file for further analysis
    with open('misrecognized_words.txt', 'w') as f:
        for word in misrecognized_words:
            f.write(f"Word ID: {word['word_id']}\n")
            f.write(f"Full Line: {word['full_line']}\n")
            f.write(f"True Label: {word['true_label']}\n")
            f.write(f"Predicted Label: {word['predicted_label']}\n")
            f.write(f"Word Error Rate: {word['wer']}\n")
            f.write(f"Character Error Rate: {word['cer']}\n")
            f.write('---\n')

