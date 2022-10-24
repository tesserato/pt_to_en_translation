import spacy
import numpy as np
import os
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import tensorflow_datasets as tfds

nlpPt = spacy.load("pt_core_news_lg")
nlpEn = spacy.load("en_core_web_lg")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running in {DEVICE} mode")
print(f"len pt vocab = {len(nlpPt.vocab)}, len en vocab = {len(nlpEn.vocab)}")

def tensorsFromDataset(train_examples):
    MAX_TRAIN_EXAMPLES = len(train_examples)
    data = []
    max_row_len = 0
    for batch, (pt_examples, en_examples) in enumerate(train_examples.batch(MAX_TRAIN_EXAMPLES).take(1)):
        for phrase, (pt, en) in enumerate(zip(pt_examples.numpy(), en_examples.numpy())):
            pt_utf8 = nlpPt(pt.decode('utf-8'))
            pt_vectors = []
            for pt_token in pt_utf8:
                pt_vectors.append(pt_token.vector)
            if len(pt_vectors) > max_row_len:
                max_row_len = len(pt_vectors)

            en_utf8 = nlpEn(en.decode('utf-8'))
            en_vectors = []
            for en_token in en_utf8:
                en_vectors.append(en_token.vector)
            if len(en_vectors) > max_row_len:
                max_row_len = len(en_vectors)

            data.append((np.array(pt_vectors), np.array(en_vectors)))
            print(
                f"creating dataloader vectors... {phrase + 1} of {MAX_TRAIN_EXAMPLES}, max len = {max_row_len}")

    pt_tensor = []
    en_tensor = []
    for pt_row, en_row in data:
        pt_row = torch.tensor(pt_row)
        print(pt_row.size())
        pt_row = torch.cat((pt_row, torch.zeros(
            max_row_len - pt_row.size()[0], 300)), 0)
        print(pt_row.size())
        # exit()
        pt_tensor.append(pt_row)

        en_row = torch.tensor(en_row)
        en_row = torch.cat((en_row, torch.zeros(
            max_row_len - en_row.size()[0], 300)), 0)
        en_tensor.append(en_row)

    pt_tensor = torch.stack(pt_tensor)
    en_tensor = torch.stack(en_tensor)
    return pt_tensor, en_tensor


if os.path.isfile("en_tensor.pt"): # assumes other 3 tensors are also available
    en_tensor = torch.load("en_tensor.pt")
    pt_tensor = torch.load("pt_tensor.pt")
    en_tensor_val = torch.load("en_tensor_val.pt")
    pt_tensor_val = torch.load("pt_tensor_val.pt")
else: # if not, creates tensors from the dataset ans saves them
    '''This will download the dataset on first run'''
    examples, metadata = tfds.load(
        'ted_hrlr_translate/pt_to_en',
        with_info=True,
        as_supervised=True
    )
    train_examples, val_examples = examples['train'], examples['validation']

    
    en_tensor, pt_tensor = tensorsFromDataset(examples['train'])
    torch.save(pt_tensor, 'pt_tensor.pt')
    torch.save(en_tensor, 'en_tensor.pt')

    en_tensor_val, pt_tensor_val = tensorsFromDataset(examples['validation'])
    torch.save(pt_tensor_val, 'pt_tensor_val.pt')
    torch.save(en_tensor_val, 'en_tensor_val.pt')


print(pt_tensor.size(), en_tensor.size())
BATCH_SIZE = 64
train = torch.utils.data.TensorDataset(pt_tensor, en_tensor)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
val = torch.utils.data.TensorDataset(pt_tensor_val, en_tensor_val)
val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)

EPOCHS = 1000


d_model = 300
transformer = Transformer(
    num_encoder_layers=4,
    num_decoder_layers=4,
    d_model=d_model,
    nhead=10,
    #  src_vocab_size=len(nlpPt.vocab),
    #  tgt_vocab_size=len(nlpEn.vocab),
)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )

for epoch in range(EPOCHS):
    for src, tgt in train_loader:
        out = transformer.forward(src, tgt)
        loss = loss_fn(out, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss.float())

exit()
for batch, (pt_examples, en_examples) in enumerate(train_examples.batch(BATCH_SIZE).take(EPOCHS)):
    transformer.train()
    print(
        f"########################## Batch {batch + 1} ##########################")
    for phrase, (pt, en) in enumerate(zip(pt_examples.numpy(), en_examples.numpy())):
        src = nlpPt(pt.decode('utf-8')).vector
        tgt = nlpEn(en.decode('utf-8')).vector
        # print(src.shape, tgt.shape)
        m = min(len(src), len(tgt))
        src = torch.tensor(src[:d_model]).unsqueeze(0).unsqueeze(0)
        tgt = torch.tensor(tgt[:d_model]).unsqueeze(0).unsqueeze(0)
        # print(src.shape, tgt.shape)
        out = transformer.forward(src, tgt)
        loss = loss_fn(out, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss.float())
    for batch, (pt_examples, en_examples) in enumerate(val_examples.batch(1).take(1)):
        transformer.eval()
        for phrase, (pt, en) in enumerate(zip(pt_examples.numpy(), en_examples.numpy())):
            pt_utf8 = pt.decode('utf-8')
            en_utf8 = en.decode('utf-8')
            print(f"pt: {pt_utf8}")
            print(f"en: {en_utf8}")
            src = nlpPt(pt_utf8).vector
            tgt = nlpEn(en_utf8).vector
            # print(src.shape, tgt.shape)
            src = torch.tensor(src[:d_model]).unsqueeze(0).unsqueeze(0)
            tgt = torch.tensor(tgt[:d_model]).unsqueeze(0).unsqueeze(0)
            # print(out.shape)

            out = transformer.forward(src, tgt)
            ms = nlpEn.vocab.vectors.most_similar(out.detach().numpy()[0], n=3)
            print(ms)
            words = [nlpEn.vocab.strings[w] for w in ms[0][0]]
            print(words)
            # print(out)


# print(f"Phrase {phrase} pt: {pt_utf8}")
# print(nlpPt(pt_utf8).vector)
# print(",".join([f"{token.text}:{token.idx}" for token in nlpPt(pt_utf8)]))
# print(f"Phrase {phrase} en: {en_utf8}")
# print(",".join([f"{token.text}:{token.idx}" for token in nlpEn(en_utf8)]))
# print()

# src = torch.rand((10, 32, 512))
# tgt = torch.rand((20, 32, 512))
# for epoch in range(10):
#   out = transformer.forward(src, tgt)
#   loss = loss_fn(out, tgt)
#   optimizer.zero_grad()
#   loss.backward()
#   optimizer.step()
#   print(loss)
# exit()
