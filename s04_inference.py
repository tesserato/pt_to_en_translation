from s02_transformerDefinition import *

transformer = torch.load("transformer.pt")
transformer.eval()


IDX = 2
print(">> teste de tradução:")
ipt_pt = src_vocab.lookup_tokens([id for id in src_val_tensor[IDX] if id != PAD_IDX])
ipt_pt = " ".join([t for t in ipt_pt if t not in special_symbols])
print("src:", ipt_pt)
tgt_en = tgt_vocab.lookup_tokens([id for id in tgt_val_tensor[IDX] if id != PAD_IDX])
tgt_en = " ".join([t for t in tgt_en if t not in special_symbols])
print("tgt:", tgt_en)

max_length = 100
model = transformer
input_sequence = torch.tensor([[BOS_IDX] + src_vocab(toTokens(ipt_pt.rstrip("\n"))) + [EOS_IDX]])

y_input = torch.tensor([[BOS_IDX]], dtype=torch.long, device=DEVICE)

num_tokens = input_sequence.size()[0]

idxs = []
for _ in range(max_length):
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(input_sequence, y_input)

    logits = transformer(input_sequence, y_input, None, None, None, None, None)
    # logits = transformer(input_sequence, y_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

    # print(logits[0].size())
    # print(torch.argmax(logits[0], 1))
    next_item = logits.topk(1)[1].view(-1)[-1].item() # num with highest probability
    next_item = torch.tensor([[next_item]], device=DEVICE)
    idxs.append(next_item.item())
    # print(idxs[-1])

    # Concatenate previous input with predicted best word
    y_input = torch.cat((y_input, next_item), dim=1)

    # Stop if model predicts end of sentence
    if next_item.view(-1).item() == EOS_IDX:
      break



tokens = tgt_vocab.lookup_tokens(idxs)
# print(tokens)
string = " ".join([t for t in tokens if t not in special_symbols])
print("inf:", string)