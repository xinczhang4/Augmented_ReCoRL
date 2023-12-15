import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForNextSentencePrediction, AdamW
import pandas as pd

class RocStoriesDataset(Dataset):
    def __init__(self, filename, tokenizer, max_len):
        self.df = pd.read_csv(filename)  # Assuming you have converted the dataset image to a CSV file
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        story = self.df.iloc[idx]
        # Combine the first four sentences as input text
        input_text = f"{story['sentence1']} {story['sentence2']} {story['sentence3']} {story['sentence4']}"
        # The fifth sentence is the label
        label_text = story['sentence5']
        inputs = self.tokenizer.encode_plus(input_text, label_text, add_special_tokens=True,
                                            max_length=self.max_len, padding='max_length',
                                            truncation=True, return_tensors='pt')
        return {'input_ids': inputs['input_ids'].squeeze(),
                'token_type_ids': inputs['token_type_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': torch.LongTensor([1])}  # Assuming label is always the correct next sentence


def train(model, dataloader, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(dataloader)}")

# Parameters
max_len = 128  # or a suitable maximum length for the sentences
batch_size = 64
learning_rate = 2e-5
epochs = 10  # or however many you deem necessary

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Load Dataset
roc_stories_dataset = RocStoriesDataset(filename='ROCStories.csv', tokenizer=tokenizer, max_len=max_len)
dataloader = DataLoader(roc_stories_dataset, batch_size=batch_size, shuffle=True)

# Train
train(model, dataloader, optimizer, device, epochs)

# Save
model.save_pretrained('roc_trained')
tokenizer.save_pretrained('roc_trained')