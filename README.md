# DL_Transformer

- 2024년 4학년 2학기 딥러닝 기반 데이터 분석 Transformer 아카이빙 레포지토리입니다.

<br/><br/>

## 1. Development Environment Assign
- Python version: 3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)] 
- PyTorch version: 1.8.1+cpu 
- TorchText version: 0.9.1 
- CUDA version: 11.8
- Numpy version : 1.22.0

<br/><br/>

## 2. logic
### 2.1 tokenizer


    import spacy
    
    class Tokenizer:
    
        def __init__(self):
            self.spacy_de = spacy.load('de_core_news_sm')
            self.spacy_en = spacy.load('en_core_web_sm')
    
        def tokenize_de(self, text):
            """
            Tokenizes German text from a string into a list of strings
            """
            return [tok.text for tok in self.spacy_de.tokenizer(text)]
    
        def tokenize_en(self, text):
            """
            Tokenizes English text from a string into a list of strings
            """
            return [tok.text for tok in self.spacy_en.tokenizer(text)]


<br/><br/>
### 2.2 data loader

    from torchtext.legacy.data import Field, BucketIterator
    from torchtext.legacy.datasets import Multi30k
    
    class DataLoader:
        source: Field = None
        target: Field = None
    
        def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
            self.ext = ext
            self.tokenize_en = tokenize_en
            self.tokenize_de = tokenize_de
            self.init_token = init_token
            self.eos_token = eos_token
            print('dataset initializing start')
    
        def make_dataset(self):
            if self.ext == ('.de', '.en'):
                self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                    lower=True, batch_first=True)
                self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                    lower=True, batch_first=True)
    
            elif self.ext == ('.en', '.de'):
                self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                    lower=True, batch_first=True)
                self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                    lower=True, batch_first=True)
    
            train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
            return train_data, valid_data, test_data
    
        def build_vocab(self, train_data, min_freq):
            self.source.build_vocab(train_data, min_freq=min_freq)
            self.target.build_vocab(train_data, min_freq=min_freq)
    
        def make_iter(self, train, validate, test, batch_size, device):
            train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                                  batch_size=batch_size,
                                                                                  device=device)
            print('dataset initializing done')
            return train_iterator, valid_iterator, test_iterator


### 2.3 data

    from conf import *
    from util.data_loader import DataLoader
    from util.tokenizer import Tokenizer
    
    tokenizer = Tokenizer()
    loader = DataLoader(ext=('.en', '.de'),
                        tokenize_en=tokenizer.tokenize_en,
                        tokenize_de=tokenizer.tokenize_de,
                        init_token='<sos>',
                        eos_token='<eos>')
    
    train, valid, test = loader.make_dataset()
    loader.build_vocab(train_data=train, min_freq=2)
    train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                         batch_size=batch_size,
                                                         device=device)
    
    src_pad_idx = loader.source.vocab.stoi['<pad>']
    trg_pad_idx = loader.target.vocab.stoi['<pad>']
    trg_sos_idx = loader.target.vocab.stoi['<sos>']
    
    enc_voc_size = len(loader.source.vocab)
    dec_voc_size = len(loader.target.vocab)


### 2.4 token embedding
    from torch import nn
    
    class TokenEmbedding(nn.Embedding):
        """
        Token Embedding using torch.nn
        they will dense representation of word using weighted matrix
        """
    
        def __init__(self, vocab_size, d_model):
            """
            class for token embedding that included positional information
    
            :param vocab_size: size of vocabulary
            :param d_model: dimensions of model
            """
            super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

### 2.5 positional encoding

    from torch import nn
    
    class TokenEmbedding(nn.Embedding):
        """
        Token Embedding using torch.nn
        they will dense representation of word using weighted matrix
        """
    
        def __init__(self, vocab_size, d_model):
            """
            class for token embedding that included positional information
    
            :param vocab_size: size of vocabulary
            :param d_model: dimensions of model
            """
            super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

### 2.6 layernorm

    import torch
    from torch import nn
    
    
    class LayerNorm(nn.Module):
        def __init__(self, d_model, eps=1e-12):
            super(LayerNorm, self).__init__()
            self.gamma = nn.Parameter(torch.ones(d_model))
            self.beta = nn.Parameter(torch.zeros(d_model))
            self.eps = eps
    
        def forward(self, x):
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, unbiased=False, keepdim=True)
            # '-1' means last dimension.
    
            out = (x - mean) / torch.sqrt(var + self.eps)
            out = self.gamma * out + self.beta
            return out

### 2.7 scale dot product attention
    import math
    
    from torch import nn
    
    
    class ScaleDotProductAttention(nn.Module):
        """
        compute scale dot product attention
    
        Query : given sentence that we focused on (decoder)
        Key : every sentence to check relationship with Qeury(encoder)
        Value : every sentence same with Key (encoder)
        """
    
        def __init__(self):
            super(ScaleDotProductAttention, self).__init__()
            self.softmax = nn.Softmax(dim=-1)
    
        def forward(self, q, k, v, mask=None, e=1e-12):
            # input is 4 dimension tensor
            # [batch_size, head, length, d_tensor]
            batch_size, head, length, d_tensor = k.size()
    
            # 1. dot product Query with Key^T to compute similarity
            k_t = k.transpose(2, 3)  # transpose
            score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
    
            # 2. apply masking (opt)
            if mask is not None:
                score = score.masked_fill(mask == 0, -10000)
    
            # 3. pass them softmax to make [0, 1] range
            score = self.softmax(score)
    
            # 4. multiply with Value
            v = score @ v
    
            return v, score

### 2.8 multi-head attention
    import sys
    import os
    
    # 프로젝트 루트를 PYTHONPATH에 추가
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from torch import nn
    
    from models.layers.scale_dot_product_attention import ScaleDotProductAttention
    
    
    class MultiHeadAttention(nn.Module):
    
        def __init__(self, d_model, n_head):
            super(MultiHeadAttention, self).__init__()
            self.n_head = n_head
            self.attention = ScaleDotProductAttention()
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.w_concat = nn.Linear(d_model, d_model)
    
        def forward(self, q, k, v, mask=None):
            # 1. dot product with weight matrices
            q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
    
            # 2. split tensor by number of heads
            q, k, v = self.split(q), self.split(k), self.split(v)
    
            # 3. do scale dot product to compute similarity
            out, attention = self.attention(q, k, v, mask=mask)
    
            # 4. concat and pass to linear layer
            out = self.concat(out)
            out = self.w_concat(out)
    
            # 5. visualize attention map
            # TODO : we should implement visualization
    
            return out
    
        def split(self, tensor):
            """
            split tensor by number of head
    
            :param tensor: [batch_size, length, d_model]
            :return: [batch_size, head, length, d_tensor]
            """
            batch_size, length, d_model = tensor.size()
    
            d_tensor = d_model // self.n_head
            tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
            # it is similar with group convolution (split by number of heads)
    
            return tensor
    
        def concat(self, tensor):
            """
            inverse function of self.split(tensor : torch.Tensor)
    
            :param tensor: [batch_size, head, length, d_tensor]
            :return: [batch_size, length, d_model]
            """
            batch_size, head, length, d_tensor = tensor.size()
            d_model = head * d_tensor
    
            tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
            return tensor


### 2.9 positionwise feed forward
    from torch import nn
    
    
    class PositionwiseFeedForward(nn.Module):
    
        def __init__(self, d_model, hidden, drop_prob=0.1):
            super(PositionwiseFeedForward, self).__init__()
            self.linear1 = nn.Linear(d_model, hidden)
            self.linear2 = nn.Linear(hidden, d_model)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=drop_prob)
    
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.linear2(x)
            return x

### 2.10 transformer embedding
    import sys
    import os
    
    # 프로젝트 루트를 PYTHONPATH에 추가
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from torch import nn
    
    from models.embedding.positional_encoding import PositionalEncoding
    from models.embedding.token_embeddings import TokenEmbedding
    
    from gensim.models import Word2Vec
    class TransformerEmbedding(nn.Module):
        """
        token embedding + positional encoding (sinusoid)
        positional encoding can give positional information to network
        """
    
        def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
            """
            class for word embedding that included positional information
    
            :param vocab_size: size of vocabulary
            :param d_model: dimensions of model
            super(TransformerEmbedding, self).__init__()
            """
            super(TransformerEmbedding, self).__init__()
            self.tok_emb = TokenEmbedding(vocab_size, d_model)
            self.pos_emb = PositionalEncoding(d_model, max_len, device)
            self.drop_out = nn.Dropout(p=drop_prob)
    
    
    
        def forward(self, x):
            tok_emb = self.tok_emb(x)
            pos_emb = self.pos_emb(x)
            return self.drop_out(tok_emb + pos_emb)

### 2.11 encoder layer

    import sys
    import os
    
    # 프로젝트 루트를 PYTHONPATH에 추가
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from torch import nn
    
    from models.layers.layer_norm import LayerNorm
    from models.layers.multi_head_attention import MultiHeadAttention
    from models.layers.position_wise_feed_forward import PositionwiseFeedForward
    
    
    class EncoderLayer(nn.Module):
    
        def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
            super(EncoderLayer, self).__init__()
            self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
            self.norm1 = LayerNorm(d_model=d_model)
            self.dropout1 = nn.Dropout(p=drop_prob)
    
            self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
            self.norm2 = LayerNorm(d_model=d_model)
            self.dropout2 = nn.Dropout(p=drop_prob)
    
        def forward(self, x, src_mask):
            # 1. compute self attention
            _x = x
            x = self.attention(q=x, k=x, v=x, mask=src_mask)
    
            # 2. add and norm
            x = self.dropout1(x)
            x = self.norm1(x + _x)
    
            # 3. positionwise feed forward network
            _x = x
            x = self.ffn(x)
    
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)
            return x

### 2.11 encoder
    import sys
    import os
    
    # 프로젝트 루트를 PYTHONPATH에 추가
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from torch import nn
    
    from models.blocks.encoder_layer import EncoderLayer
    from models.embedding.transformer_embedding import TransformerEmbedding
    
    
    class Encoder(nn.Module):
    
        def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
            super().__init__()
            self.emb = TransformerEmbedding(d_model=d_model,
                                            max_len=max_len,
                                            vocab_size=enc_voc_size,
                                            drop_prob=drop_prob,
                                            device=device,
                                            )
    
            self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                      ffn_hidden=ffn_hidden,
                                                      n_head=n_head,
                                                      drop_prob=drop_prob)
                                         for _ in range(n_layers)])
    
        def forward(self, x, src_mask):
            x = self.emb(x)
    
            for layer in self.layers:
                x = layer(x, src_mask)
    
            return x

### 2.12 decoder layer
    import sys
    import os
    
    # 프로젝트 루트를 PYTHONPATH에 추가
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from torch import nn
    
    from models.layers.layer_norm import LayerNorm
    from models.layers.multi_head_attention import MultiHeadAttention
    from models.layers.position_wise_feed_forward import PositionwiseFeedForward
    
    
    class DecoderLayer(nn.Module):
    
        def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
            super(DecoderLayer, self).__init__()
            self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
            self.norm1 = LayerNorm(d_model=d_model)
            self.dropout1 = nn.Dropout(p=drop_prob)
    
            self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
            self.norm2 = LayerNorm(d_model=d_model)
            self.dropout2 = nn.Dropout(p=drop_prob)
    
            self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
            self.norm3 = LayerNorm(d_model=d_model)
            self.dropout3 = nn.Dropout(p=drop_prob)
    
        def forward(self, dec, enc, trg_mask, src_mask):
            # 1. compute self attention
            _x = dec
            x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
    
            # 2. add and norm
            x = self.dropout1(x)
            x = self.norm1(x + _x)
    
            if enc is not None:
                # 3. compute encoder - decoder attention
                _x = x
                x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
    
                # 4. add and norm
                x = self.dropout2(x)
                x = self.norm2(x + _x)
    
            # 5. positionwise feed forward network
            _x = x
            x = self.ffn(x)
    
            # 6. add and norm
            x = self.dropout3(x)
            x = self.norm3(x + _x)
            return x

### 2.13 decoder
    import sys
    import os
    
    # 프로젝트 루트를 PYTHONPATH에 추가
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    import torch
    from torch import nn
    
    from models.blocks.decoder_layer import DecoderLayer
    from models.embedding.transformer_embedding import TransformerEmbedding
    
    
    class Decoder(nn.Module):
        def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
            super().__init__()
            self.emb = TransformerEmbedding(d_model=d_model,
                                            drop_prob=drop_prob,
                                            max_len=max_len,
                                            vocab_size=dec_voc_size,
                                            device=device,
                                            )
    
            self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                      ffn_hidden=ffn_hidden,
                                                      n_head=n_head,
                                                      drop_prob=drop_prob)
                                         for _ in range(n_layers)])
    
            self.linear = nn.Linear(d_model, dec_voc_size)
    
        def forward(self, trg, enc_src, trg_mask, src_mask):
            trg = self.emb(trg)
    
            for layer in self.layers:
                trg = layer(trg, enc_src, trg_mask, src_mask)
    
            # pass to LM head
            output = self.linear(trg)
            return output

### 2.14 transformer
    import sys
    import os
    
    # 프로젝트 루트를 PYTHONPATH에 추가
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    import torch
    from torch import nn
    
    from models.model.decoder import Decoder
    from models.model.encoder import Encoder
    
    
    class Transformer(nn.Module):
    
        def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                     ffn_hidden, n_layers, drop_prob, device):
            super().__init__()
            self.src_pad_idx = src_pad_idx
            self.trg_pad_idx = trg_pad_idx
            self.trg_sos_idx = trg_sos_idx
            self.device = device
            self.encoder = Encoder(d_model=d_model,
                                   n_head=n_head,
                                   max_len=max_len,
                                   ffn_hidden=ffn_hidden,
                                   enc_voc_size=enc_voc_size,
                                   drop_prob=drop_prob,
                                   n_layers=n_layers,
                                   device=device)
    
            self.decoder = Decoder(d_model=d_model,
                                   n_head=n_head,
                                   max_len=max_len,
                                   ffn_hidden=ffn_hidden,
                                   dec_voc_size=dec_voc_size,
                                   drop_prob=drop_prob,
                                   n_layers=n_layers,
                                   device=device)
    
        def forward(self, src, trg):
            src_mask = self.make_src_mask(src)
            trg_mask = self.make_trg_mask(trg)
            enc_src = self.encoder(src, src_mask)
            output = self.decoder(trg, enc_src, trg_mask, src_mask)
            return output
    
        def make_src_mask(self, src):
            src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
            return src_mask
    
        def make_trg_mask(self, trg):
            trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3).bool()
            trg_len = trg.shape[1]
            trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device).bool()
            trg_mask = trg_pad_mask & trg_sub_mask
            return trg_mask

### 2.15 train
    import math
    import time
    
    from torch import nn, optim
    from torch.optim import Adam
    
    from data import *
    from models.model.transformer import Transformer
    from util.bleu import idx_to_word, get_bleu
    from util.epoch_timer import epoch_time
    
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.kaiming_uniform(m.weight.data)
    
    
    model = Transformer(src_pad_idx=src_pad_idx,
                        trg_pad_idx=trg_pad_idx,
                        trg_sos_idx=trg_sos_idx,
                        d_model=d_model,
                        enc_voc_size=enc_voc_size,
                        dec_voc_size=dec_voc_size,
                        max_len=max_len,
                        ffn_hidden=ffn_hidden,
                        n_head=n_heads,
                        n_layers=n_layers,
                        drop_prob=drop_prob,
                        device=device).to(device)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    model.apply(initialize_weights)
    optimizer = Adam(params=model.parameters(),
                     lr=init_lr,
                     weight_decay=weight_decay,
                     eps=adam_eps)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     verbose=True,
                                                     factor=factor,
                                                     patience=patience)
    
    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
    
    
    def train(model, iterator, optimizer, criterion, clip):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
    
            optimizer.zero_grad()
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)
    
            loss = criterion(output_reshape, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
    
            epoch_loss += loss.item()
            print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
    
        return epoch_loss / len(iterator)
    
    
    def evaluate(model, iterator, criterion):
        model.eval()
        epoch_loss = 0
        batch_bleu = []
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch.src
                trg = batch.trg
                output = model(src, trg[:, :-1])
                output_reshape = output.contiguous().view(-1, output.shape[-1])
                trg = trg[:, 1:].contiguous().view(-1)
    
                loss = criterion(output_reshape, trg)
                epoch_loss += loss.item()
    
                total_bleu = []
                for j in range(batch_size):
                    try:
                        trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                        output_words = output[j].max(dim=1)[1]
                        output_words = idx_to_word(output_words, loader.target.vocab)
                        bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                        total_bleu.append(bleu)
                    except:
                        pass
    
                total_bleu = sum(total_bleu) / len(total_bleu)
                batch_bleu.append(total_bleu)
    
        batch_bleu = sum(batch_bleu) / len(batch_bleu)
        return epoch_loss / len(iterator), batch_bleu
    
    
    def run(total_epoch, best_loss):
        train_losses, test_losses, bleus = [], [], []
        for step in range(total_epoch):
            start_time = time.time()
            train_loss = train(model, train_iter, optimizer, criterion, clip)
            valid_loss, bleu = evaluate(model, valid_iter, criterion)
            end_time = time.time()
    
            if step > warmup:
                scheduler.step(valid_loss)
    
            train_losses.append(train_loss)
            test_losses.append(valid_loss)
            bleus.append(bleu)
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))
    
            f = open('result/train_loss.txt', 'w')
            f.write(str(train_losses))
            f.close()
    
            f = open('result/bleu.txt', 'w')
            f.write(str(bleus))
            f.close()
    
            f = open('result/test_loss.txt', 'w')
            f.write(str(test_losses))
            f.close()
    
            print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
            print(f'\tBLEU Score: {bleu:.3f}')
    
    
    if __name__ == '__main__':
        run(total_epoch=epoch, best_loss=inf)
