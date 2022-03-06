# from typing import Text
# from Geo_bert.train import Dataset
from transformers import BertConfig,BertTokenizerFast,BertForMaskedLM
import torch
from transformers import AdamW
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()


tokenizer = BertTokenizerFast.from_pretrained('Geobert-base')
#model = BertForMaskedLM.from_pretrained('bert-base-uncased')

with open('All-400.txt','r') as fp:
    text = fp.read().split('\n')
    #print(text[:5])
print("实例长度：",len(text))
inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation = True, padding = 'max_length')
#print(inputs)
inputs['labels'] = inputs.input_ids.detach().clone()

rand = torch.rand(inputs.input_ids.shape)

mask_arr = (rand < 0.15) * (inputs.input_ids != 1) * (inputs.input_ids != 4) * (inputs.input_ids != 0)

selection = []
for i in range(mask_arr.shape[0]):
    selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

for i in range(mask_arr.shape[0]):
    inputs.input_ids[i,selection[i]] = 4

class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key:torch.tensor(val[idx]) for key,val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

dataset = MeditationsDataset(inputs)

torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl')
sampler = DistributedSampler(dataset)

#dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32,shuffle=True,sampler = sampler)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 16,sampler = sampler)

config = BertConfig(
    vocab_size=30522,
    hidden_size=768, 
    num_hidden_layers=12, 
    num_attention_heads=12,
    max_position_embeddings=512
)
model = BertForMaskedLM(config)


model = model.cuda()  # 在使用DistributedDataParallel之前，需要先将模型放到GPU上
model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
model.train()
optim = AdamW(model.module.parameters(),lr = 1e-5)

epochs = 40

for epoch in range(epochs):
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()

        outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

model.module.save_pretrained('./Geobert-base')

