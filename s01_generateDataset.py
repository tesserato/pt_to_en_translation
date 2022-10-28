import tensorflow_datasets as tfds
'''
Run this file to download the dataset ans save it to disk as 4 .txt files
''' 

# This will download the dataset on first run
examples, metadata = tfds.load(
    'ted_hrlr_translate/pt_to_en',
    with_info=True,
    as_supervised=True
)
train_examples, val_examples = examples['train'], examples['validation']

MAX_TRAIN_EXAMPLES = len(train_examples)
data_pt = []
data_en = []
for batch, (pt_examples, en_examples) in enumerate(train_examples.batch(MAX_TRAIN_EXAMPLES).take(1)):
    for phrase, (pt, en) in enumerate(zip(pt_examples.numpy(), en_examples.numpy())):
        data_pt.append(pt.decode('utf-8') + "\n")
        data_en.append(en.decode('utf-8') + "\n")
        print(f"creating training data... {phrase + 1} of {MAX_TRAIN_EXAMPLES}")

f = open("01_train_pt.txt", "w", encoding="utf-8")
f.writelines(data_pt)
f.close()

f = open("02_train_en.txt", "w", encoding="utf-8")
f.writelines(data_en)
f.close()

MAX_VAL_EXAMPLES = len(val_examples)
data_pt = []
data_en = []
for batch, (pt_examples, en_examples) in enumerate(val_examples.batch(MAX_VAL_EXAMPLES).take(1)):
    for phrase, (pt, en) in enumerate(zip(pt_examples.numpy(), en_examples.numpy())):
        data_pt.append(pt.decode('utf-8') + "\n")
        data_en.append(en.decode('utf-8') + "\n")
        print(f"creating validation data... {phrase + 1} of {MAX_VAL_EXAMPLES}")

f = open("03_val_pt.txt", "w", encoding="utf-8")
f.writelines(data_pt)
f.close()

f = open("04_val_en.txt", "w", encoding="utf-8")
f.writelines(data_en)
f.close()