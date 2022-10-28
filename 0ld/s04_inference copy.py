from transformerDefinition import *


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

model = transformer
model.eval()

src = torch.tensor([[BOS_IDX] + src_vocab(toTokens(ipt_pt.rstrip("\n"))) + [EOS_IDX]]).to(DEVICE)#.view(-1, 1)
print(src)
num_tokens = src.size()[0]
print(num_tokens)
src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(DEVICE)
# src = src
# src_mask = src_mask
# start_symbol=BOS_IDX
memory = model.encode(src, src_mask).to(DEVICE)
ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(DEVICE)
for i in range(num_tokens + 4):
    # memory = memory
    tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
    out = model.decode(ys, memory, tgt_mask)
    # out = out.transpose(0, 1)
    prob = model.generator(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    next_word = next_word.item()

    ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
    if next_word == EOS_IDX:
        break


print(
  "inf:", " ".join(tgt_vocab.lookup_tokens(list(ys.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
)
