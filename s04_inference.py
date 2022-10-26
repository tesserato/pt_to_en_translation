from s02_transformerDefinition import *

transformer = torch.load("transformer.pt")
transformer.eval()

raw_pt = "mas e se estes fatores fossem ativos ?"
max_length = 100
model = transformer
input_sequence = torch.tensor([[BOS_IDX] + src_vocab(toTokens(raw_pt.rstrip("\n"))) + [EOS_IDX]])

y_input = torch.tensor([[BOS_IDX]], dtype=torch.long, device=DEVICE)

num_tokens = input_sequence.size()[0]

idxs = []
for _ in range(max_length):
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(input_sequence, y_input)

    # logits = transformer(input_sequence, y_input, None, None, None, None, None)
    logits = transformer(input_sequence, y_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
    print(logits[0].size())
    print(torch.argmax(logits[0], 1))
    next_item = logits.topk(1)[1].view(-1)[-1].item() # num with highest probability
    next_item = torch.tensor([[next_item]], device=DEVICE)
    idxs.append(next_item.item())
    print(idxs[-1])

    # Concatenate previous input with predicted best word
    y_input = torch.cat((y_input, next_item), dim=1)

    # Stop if model predicts end of sentence
    if next_item.view(-1).item() == EOS_IDX:
      break


print(">> teste de tradução:")
tokens = tgt_vocab.lookup_tokens(idxs)
print(tokens)
string = " ".join([t for t in tokens if t not in special_symbols])
print(string)