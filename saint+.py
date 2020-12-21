import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split 

import gc
import copy

# CONFIG

device = "cpu"
MAX_SEQ = 100
EMBED_DIMS = 128
ENC_HEADS = DEC_HEADS = 8
NUM_ENCODER = NUM_DECODER = 6
BATCH_SIZE = 32
TRAIN_FILE = "../Riiid_kaggle/riiid-test-answer-prediction/train.csv"
TOTAL_EXE = 13523
TOTAL_CAT = 13523

EPOCH = 20

# Data Loader

class DKTDataset(Dataset):
  def __init__(self, group, n_skills, max_seq=100):
    self.samples = group
    self.n_skills = n_skills
    self.max_seq = max_seq
    self.data = []

    for que, ans, res_time, lag_time, exe_cat in self.samples:
        if len(que)>=self.max_seq: # extent 0 to the front
            self.data.extend([(que[l:l + self.max_seq],
                               ans[l:l + self.max_seq],
                               res_time[l:l + self.max_seq],
                               lag_time[l:l + self.max_seq],
                               exe_cat[l:l + self.max_seq])\
            for l in range(len(que)) if l % self.max_seq==0])
        elif len(que)<self.max_seq and len(que)>10: # drop sample that has less than 10 questions
            self.data.append((que, ans, res_time, lag_time, exe_cat))
        else :
            continue
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    content_ids, answered_correctly, response_time, lag_time, exe_category = self.data[idx]
    seq_len = len(content_ids)

    q_ids = np.zeros(self.max_seq, dtype=int)
    ans = np.zeros(self.max_seq, dtype=int)
    r_time = np.zeros(self.max_seq, dtype=int)
    l_time = np.zeros(self.max_seq, dtype=int)
    exe_cat = np.zeros(self.max_seq, dtype=int)

    if seq_len >= self.max_seq:
      q_ids[:] = content_ids[-self.max_seq:]
      ans[:] = answered_correctly[-self.max_seq:]
      r_time[:] = response_time[-self.max_seq:]
      l_time[:] = lag_time[-self.max_seq:]
      exe_cat[:] = exe_category[-self.max_seq:]
    else:
      q_ids[-seq_len:] = content_ids
      ans[-seq_len:] = answered_correctly
      r_time[-seq_len:] = response_time
      l_time[-seq_len:] = lag_time
      exe_cat[-seq_len:] = exe_category
    
    target_qids = q_ids[1:]
    label = ans[1:] 

    input_ids = np.zeros(self.max_seq-1,dtype=int)
    input_ids = q_ids[:-1].copy()

    input_rtime = np.zeros(self.max_seq-1,dtype=int)
    input_rtime = r_time[:-1].copy()
    
    input_ltime = np.zeros(self.max_seq-1, dtype=int)
    input_ltime = l_time[:-1].copy()

    input_cat = np.zeros(self.max_seq-1,dtype=int)
    input_cat = exe_cat[:-1].copy()

    _input = {"input_ids": input_ids,
              "input_rtime": input_rtime.astype(np.int),
              "input_ltime": input_ltime.astype(np.int),
              "input_cat": input_cat}

    return _input, target_qids, label 


def get_lt(grouped_x):
    # calclate elapsed time and lag time for current question
    time_sort_df = grouped_x.sort_values(['timestamp'], ascending=True)
    time_sort_df.prior_question_elapsed_time = time_sort_df.prior_question_elapsed_time.shift(-1)

#     time_sort_df.prior_question_had_explanation = time_sort_df.prior_question_had_explanation.shift(-1)
#     time_sort_df.prior_question_had_explanation.fillna(0, inplace=True)

    time_sort_df["lag_time"] = time_sort_df["timestamp"].diff()
    return time_sort_df



def get_dataloaders():              
    dtypes = {'timestamp': 'int64', 'user_id': 'int32' ,'content_id': 'int16',
                'answered_correctly':'int8',"prior_question_elapsed_time":"float32","task_container_id":"int16"}
    print("loading csv.....")
    train_df = pd.read_csv(TRAIN_FILE,dtype=dtypes, nrows=1000000)
    #train_df = pd.read_csv(TRAIN_FILE,dtype=dtypes)
    print("shape of dataframe :",train_df.shape) 

    train_df = train_df[train_df.content_type_id==0]  # only consider question
    
    train_df = train_df.groupby('user_id', as_index=False).apply(get_lt)
    train_df.index = train_df.index.droplevel(0)
    train_df.sort_index()
    
    train_df.prior_question_elapsed_time /= 1000  # ms to s
    train_df.lag_time /= (1000 * 60) # ms to min
    train_df.prior_question_elapsed_time.fillna(300,inplace=True)
    train_df.lag_time.fillna(1440, inplace=True)
    train_df.prior_question_elapsed_time.clip(lower=0,upper=300,inplace=True) # max 300 s
    train_df.lag_time.clip(lower=0, upper=1440, inplace=True)
    train_df.prior_question_elapsed_time = train_df.prior_question_elapsed_time.astype(np.int)
    train_df.lag_time = train_df.lag_time.astype(np.int)
    
    
    train_df = train_df.sort_values(["timestamp"],ascending=True).reset_index(drop=True)
    skills = train_df.content_id.unique()
    n_skills = len(skills)
    n_cats = len(train_df.task_container_id.unique())+100
    print("no. of skills :",n_skills)
    print("no. of categories: ", n_cats)
    print("shape after exlusion:",train_df.shape)

    # grouping based on user_id to get the data supplu
    print("Grouping users...")
    group = train_df[["user_id","content_id","answered_correctly","prior_question_elapsed_time", "lag_time", "task_container_id"]]\
                    .groupby("user_id")\
                    .apply(lambda r: (r.content_id.values,
                                      r.answered_correctly.values,
                                      r.prior_question_elapsed_time.values,
                                      r.lag_time.values,
                                      r.task_container_id.values))
    del train_df
    gc.collect()

    print("splitting")
    train,val = train_test_split(group, test_size=0.2) 
    print("train size: ",train.shape,"validation size: ",val.shape)
    train_dataset = DKTDataset(train.values, n_skills=n_skills, max_seq=MAX_SEQ)
    val_dataset = DKTDataset(val.values, n_skills=n_skills, max_seq=MAX_SEQ)
    train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=2,
                          shuffle=True)
    val_loader = DataLoader(val_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=2,
                          shuffle=False)
    del train_dataset,val_dataset
    gc.collect()
    return train_loader, val_loader


# Util Function
def ut_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool)

def lt_mask(seq_len):
    """ Lower Triangular Mask
    """
    return torch.tril(torch.ones(seq_len,seq_len),diagonal=-1).to(dtype=torch.bool)

def pos_encode(seq_len):
    """ position Encoding
    """
    return torch.arange(seq_len).unsqueeze(0)

def get_clones(module, N):
    """ Cloning nn modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# Mutilhead FNN
class FFN(nn.Module):
  def __init__(self,features):
    super(FFN,self).__init__()
    self.layer1 = nn.Linear(features, features)
    self.layer2 = nn.Linear(features, features)
    self.relu = nn.ReLU()
    self.drop = nn.Dropout(0.2)
    
  def forward(self, x):
    out = self.drop(self.relu(self.layer1(x)))
    out = self.layer2(out)
    return out

class MultiHeadWithFFN(nn.Module):
  def __init__(self,n_heads,n_dims,mask_type="ut",dropout=0.2):
    super(MultiHeadWithFFN,self).__init__()
    self.n_dims = n_dims
    self.mask_type = mask_type
    self.multihead_attention = nn.MultiheadAttention(embed_dim = n_dims,
                                                      num_heads = n_heads,
                                                        dropout = dropout)
    self.layer_norm1 = nn.LayerNorm(n_dims)
    self.ffn = FFN(features = n_dims)
    self.layer_norm2 = nn.LayerNorm(n_dims)


  def forward(self,q_input,kv_input):
    q_input = q_input.permute(1,0,2)
    kv_input = kv_input.permute(1,0,2)
    query_norm = self.layer_norm1(q_input)
    kv_norm = self.layer_norm1(kv_input)
    if self.mask_type=="ut":
      mask = ut_mask(q_input.size(0))
    else: 
      mask = lt_mask(q_input.size(0))
    if device == "cuda":
      mask = mask.cuda()
    out_atten , weights_attent = self.multihead_attention(query=query_norm,
                                                key = kv_norm,
                                                value = kv_norm,
                                                attn_mask = mask)
    out_atten +=  query_norm
    out_atten = out_atten.permute(1,0,2)
    output_norm = self.layer_norm2(out_atten)
    output = self.ffn(output_norm)
    return output + output_norm 


# SAINT model
class SAINT(nn.Module):
    def __init__(self,n_encoder,n_decoder,enc_heads,dec_heads,n_dims,total_ex,total_cat,total_responses,seq_len):
        super(SAINT,self).__init__()
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.enocder = get_clones(EncoderBlock(enc_heads,n_dims,total_ex,total_cat,seq_len),n_encoder)
        self.decoder = get_clones(DecoderBlock(dec_heads,n_dims,total_responses,seq_len),n_decoder)
        self.fc = nn.Linear(n_dims,1)
    
    def forward(self, in_exercise, in_category, in_response, in_etime, in_ltime):
        first_block = True
        for n in range(self.n_encoder):
          if n>=1:
            first_block=False
          
          enc = self.enocder[n](in_exercise, in_category, first_block=first_block)
          in_exercise = enc
          in_category = enc

        first_block = True
        for n in range(self.n_decoder):
          if n>=1:
            first_block=False
          dec = self.decoder[n](in_response, in_exercise, in_etime, in_ltime, first_block=first_block)
          in_exercise = dec
          in_response = dec
          in_etime = dec
          in_ltime = dec
          
        return torch.sigmoid(self.fc(dec))



class EncoderBlock(nn.Module):
  def __init__(self,n_heads,n_dims,total_ex,total_cat,seq_len):
    super(EncoderBlock,self).__init__()
    self.seq_len = seq_len
    self.exercise_embed = nn.Embedding(total_ex,n_dims)
    self.category_embed = nn.Embedding(total_cat,n_dims)
    self.position_embed = nn.Embedding(seq_len,n_dims)
    self.layer_norm = nn.LayerNorm(n_dims)
    
    self.multihead = MultiHeadWithFFN(n_heads=n_heads,
                                            n_dims = n_dims)
  
  def forward(self, input_e, category,first_block=True):
    if first_block:
      _exe = self.exercise_embed(input_e)
      _cat = self.category_embed(category)
      position_encoded = pos_encode(self.seq_len-1)
      if device == "cuda":
        position_encoded = position_encoded.cuda()
      _pos = self.position_embed(position_encoded)
      out = _cat + _exe + _pos  # sum of exercise id, exercise category and position
    else:
      out = input_e
    output = self.multihead(q_input=out,kv_input=out)
    return output


class DecoderBlock(nn.Module):
    def __init__(self, n_heads, n_dims, total_responses, seq_len):
      super(DecoderBlock,self).__init__()
      self.seq_len = seq_len
      self.response_embed = nn.Embedding(total_responses, n_dims)
      self.etime_embed = nn.Embedding(301, n_dims)
      self.ltime_embed = nn.Embedding(1441, n_dims)
      self.position_embed = nn.Embedding(seq_len, n_dims)
      self.layer_norm = nn.LayerNorm(n_dims)
      self.multihead_attention = nn.MultiheadAttention(embed_dim=n_dims,
                                            num_heads = n_heads,
                                            dropout = 0.2)
      self.multihead = MultiHeadWithFFN(n_heads=n_heads,
                                              n_dims = n_dims)

    def forward(self, input_r, encoder_output, e_time, l_time, first_block=True):
      if first_block:
        _response = self.response_embed(input_r)
        _e_time = self.etime_embed(e_time)
        _l_time = self.ltime_embed(l_time)
        position_encoded = pos_encode(self.seq_len-1)
        if device == "cuda":
          position_encoded = position_encoded.cuda()
        _pos = self.position_embed(position_encoded)
        out = _response + _e_time + _l_time + _pos  # sum of response and position
      else:
        out = input_r      
      out = out.permute(1, 0, 2)    
      #assert out_embed.size(0)==n_dims, "input dimention should be (seq_len,batch_size,dims)"
      out_norm = self.layer_norm(out)
      mask = ut_mask(out_norm.size(0))
      if device == "cuda":
        mask = mask.cuda()
      out_atten , weights_attent = self.multihead_attention(query=out_norm,
                                                  key = out_norm,
                                                  value = out_norm,
                                                  attn_mask = mask)
      out_atten +=  out_norm
      out_atten = out_atten.permute(1,0,2)
      output = self.multihead(q_input=out_atten,kv_input=encoder_output)
      return output


class SAINTModel(pl.LightningModule):
  def __init__(self, model_args):
    super().__init__()
    self.model = SAINT(**model_args)
      
  
  def forward(self, exercise, category, response, etime, ltime):
    return self.model(exercise, category, response, etime, ltime)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(),lr=1e-3)
  
  def training_step(self, batch, batch_idx):
    inputs, target_ids, target = batch
    output = self(inputs["input_ids"], inputs["input_cat"],
                  target_ids, inputs["input_rtime"], inputs["input_ltime"])
    target_mask = (target_ids != 0)
    output = torch.masked_select(output.squeeze(), target_mask)
    target = torch.masked_select(target, target_mask)
    loss = nn.BCEWithLogitsLoss()(output.float(), target.float())
    return {"loss":loss,"output":output,"target":target}
  
  def validation_step(self,batch,batch_idx):
    inputs,target_ids,target = batch
    output = self(inputs["input_ids"], inputs["input_cat"],
                  target_ids, inputs["input_rtime"], inputs["input_ltime"])
    target_mask = (target_ids != 0)
    output = torch.masked_select(output.squeeze(),target_mask)
    target = torch.masked_select(target,target_mask)
    loss = nn.BCEWithLogitsLoss()(output.float(),target.float())
    auc = roc_auc_score(target.float(), output.float())
    return {"val_loss":loss,"output":output,"target":target, "auc":auc}

train_loader, val_loader = get_dataloaders()

ARGS = {"n_dims":EMBED_DIMS ,
            'n_encoder':NUM_ENCODER,
            'n_decoder':NUM_DECODER,
            'enc_heads':ENC_HEADS,
            'dec_heads':DEC_HEADS,
            'total_ex':TOTAL_EXE,
            'total_cat':TOTAL_CAT,
            'total_responses':TOTAL_EXE,
            'seq_len':MAX_SEQ}

########### TRAINING AND SAVING MODEL #######
checkpoint = ModelCheckpoint(filename="{epoch}_model",
                              verbose=True,
                              save_top_k=1,
                              monitor="auc")

saint_model = SAINTModel(model_args=ARGS)
trainer = pl.Trainer(progress_bar_refresh_rate=21,
                      max_epochs=EPOCH, callbacks=[checkpoint]) 
trainer.fit(model=saint_model,
              train_dataloader=train_loader,val_dataloaders=val_loader) 
trainer.save_checkpoint("model_saint.pt")


