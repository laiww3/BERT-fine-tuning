import tokenizers
import os
bwpt = tokenizers.BertWordPieceTokenizer(unk_token='[UNK]',sep_token='[SEP]',cls_token='[CLS]',wordpieces_prefix='##')
 

bwpt.train(
    "All-400.txt",
    vocab_size=30522,
    min_frequency=3,
    limit_alphabet=1000,
    special_tokens=[
                   '[PAD]','[CLS]','[UNK]','[MASK]','[SEP]'
               ]
    
   
)
os.mkdir('Geobert-base')
bwpt.save_model( 'Geobert-base')