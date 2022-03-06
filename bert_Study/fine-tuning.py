from transformers import BertTokenizer, BertForPreTraining
import torch
from transformers import AdamW
from torch.utils.data.distributed import DistributedSampler
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained('bert-fine-tuning-120')
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased')

with open('All-400.txt','r') as fp:
    text = fp.read().split('\n')

bag = [item for sentence in text for item in sentence.split('.') if item != '']
bag_size = len(bag)

import random

sentence_a = []
sentence_b = []
label = []

for paragraph in text:
    sentences = [
        sentence for sentence in paragraph.split('.') if sentence != ''
    ]
    num_sentences = len(sentences)
    if num_sentences > 1:
        start = random.randint(0, num_sentences-2)
        # 50/50 whether is IsNextSentence or NotNextSentence
        if random.random() >= 0.5:
            # this is IsNextSentence
            sentence_a.append(sentences[start])
            sentence_b.append(sentences[start+1])
            label.append(0)
        else:
            index = random.randint(0, bag_size-1)
            # this is NotNextSentence
            sentence_a.append(sentences[start])
            sentence_b.append(bag[index])
            label.append(1)
for i in range(3):
    print(label[i])
    print(sentence_a[i] + '\n---')
    print(sentence_b[i] + '\n')

inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt',
                   max_length=512, truncation=True, padding='max_length')
inputs['next_sentence_label'] = torch.LongTensor([label]).T

print(inputs.next_sentence_label[:10])

inputs['labels'] = inputs.input_ids.detach().clone()

# create random array of floats with equal dimensions to input_ids tensor
rand = torch.rand(inputs.input_ids.shape)
# create mask array
mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
           (inputs.input_ids != 102) * (inputs.input_ids != 0)

selection = []

for i in range(inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )

for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]] = 103
    

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)
    
dataset = OurDataset(inputs)

#loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl')
sampler = DistributedSampler(dataset)

#dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32,shuffle=True,sampler = sampler)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 16,sampler = sampler)

model = model.cuda()  # 在使用DistributedDataParallel之前，需要先将模型放到GPU上
model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
model.train()

from tqdm import tqdm  # for our progress bar
optim = AdamW(model.module.parameters(),lr = 1e-5)

epochs = 120

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].cuda()
        token_type_ids = batch['token_type_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        next_sentence_label = batch['next_sentence_label'].cuda()
        labels = batch['labels'].cuda()
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        next_sentence_label=next_sentence_label,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
model.module.save_pretrained('./bert-fine-tuning-120')










