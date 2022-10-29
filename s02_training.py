'''
runs the training process.
'''
from transformerDefinition import *
import os
import numpy as np

ROOT = ""

torch.manual_seed(0)
EPOCHS = 1000
SRC_VOCAB_SIZE = len(src_vocab)
TGT_VOCAB_SIZE = len(tgt_vocab)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 8 #128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

train = torch.utils.data.TensorDataset(src_train_tensor, tgt_train_tensor)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
val = torch.utils.data.TensorDataset(src_val_tensor, tgt_val_tensor)
val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)

PATH_TO_STATE_DICT = ROOT + "transformer state_dict.pt"
PATH_TO_BEST_LOSS = ROOT + "bestLoss.txt"

bestEvalLoss = np.inf
if os.path.isfile(PATH_TO_STATE_DICT):
  bestEvalLoss = np.genfromtxt(PATH_TO_BEST_LOSS)
  tsd = torch.load(PATH_TO_STATE_DICT)
  transformer = Seq2SeqTransformer(
    tsd["NUM_ENCODER_LAYERS"], tsd["NUM_DECODER_LAYERS"], tsd["EMB_SIZE"], tsd["NHEAD"], tsd["SRC_VOCAB_SIZE"], tsd["TGT_VOCAB_SIZE"], tsd["FFN_HID_DIM"]
  )
  transformer.load_state_dict(tsd['model_state_dict'])
else:
  transformer = Seq2SeqTransformer(
    NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM
  )
  for p in transformer.parameters():
      if p.dim() > 1:
          nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


#########################


for epoch in range(EPOCHS):
  print(f"#### Epoch {epoch} ####")
  transformer.train()
  for iter, (src, tgt) in enumerate(train_loader):
      src = src.to(DEVICE)
      tgt = tgt.to(DEVICE)
      tgt_input = tgt[:, :-1]

      # print("ipt sizes", src.size(), tgt_input.size())

      src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

      logits = transformer(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
      # print(src.size(), logits.size(), len(tgt_vocab))
      optimizer.zero_grad()

      tgt_out = tgt[:, 1:]
      loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
      loss.backward()
      optimizer.step()
      print(loss.item())
      if (iter + 1) % 5 == 0:
        print("Evaluating:")
        transformer.eval()
        totalEvalLoss = []
        for iterVal, (src, tgt) in enumerate(val_loader):
          src = src.to(DEVICE)
          tgt = tgt.to(DEVICE)
          tgt_input = tgt[:, :-1]
          # print("ipt sizes", src.size(), tgt_input.size())
          
          src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
          logits = transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
          optimizer.zero_grad()

          tgt_out = tgt[:, 1:]
          loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
          loss.backward()
          optimizer.step()
          totalEvalLoss.append(loss.item())
          if iterVal > 5:
            break
        evalLoss = np.average(totalEvalLoss)
        print(f"    Eval average loss = {evalLoss}")
        if evalLoss < bestEvalLoss:
          print(f"    average validation loss improved from {np.round(bestEvalLoss, 3)} to {np.round(evalLoss, 3)} ")
          # # Method 1: saving model directly
          # torch.save(transformer, pathToTransformer + ".pt") 
          # Method 2: saving state_dict and model parameters
          torch.save(
            {
              "model_state_dict": transformer.state_dict(),
              "NUM_ENCODER_LAYERS":NUM_ENCODER_LAYERS,
              "NUM_DECODER_LAYERS":NUM_DECODER_LAYERS,
              "EMB_SIZE":EMB_SIZE,
              "NHEAD":NHEAD,
              "SRC_VOCAB_SIZE":SRC_VOCAB_SIZE,
              "TGT_VOCAB_SIZE":TGT_VOCAB_SIZE,
              "FFN_HID_DIM":FFN_HID_DIM
            }
            , PATH_TO_STATE_DICT
          )

          bestEvalLoss = evalLoss
          np.savetxt(PATH_TO_BEST_LOSS, [bestEvalLoss], fmt='%f')