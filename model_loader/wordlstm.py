import utils as util

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class WordLSTM(nn.Module):

    def __init__(self, embedding, hidden_size=150, depth=1, dropout=0.3, batch_size = 32, nclasses=2, config = None):
        super(WordLSTM, self).__init__()
        self.config = config
        self.batch_size = batch_size

        self.drop = nn.Dropout(dropout)
        self.emb_layer = EmbeddingLayer(
            embs = util.load_embedding(embedding)
        )
        self.word2id = self.emb_layer.word2id

        self.encoder = nn.LSTM(
            self.emb_layer.n_d,
            hidden_size//2,
            depth,
            dropout = dropout,
            # batch_first=True,
            bidirectional=True
        )
        d_out = hidden_size
        self.out = nn.Linear(d_out, nclasses)

    def forward(self, input):
        emb = self.emb_layer(input)
        emb = self.drop(emb)
        output, hidden = self.encoder(emb)
        output = torch.max(output, dim=0)[0].squeeze()
        output = self.drop(output)
        return self.out(output)

    def text_pred(self, text):
        batches_x = util.create_batches_x(
            text,
            self.batch_size,
            self.word2id
        )
        outs = []
        with torch.no_grad():
            for x in batches_x:
                x = Variable(x)
                emb = self.emb_layer(x)
                output, hidden = self.encoder(emb)
                output = torch.max(output, dim=0)[0]
                outs.append(F.softmax(self.out(output), dim=-1))

        return torch.cat(outs, dim=0)

    def _gpu_init(self):
        self.use_gpu = False
        self.device_ids = [0]
        self.device = torch.device('cpu')
        if self.config.GPU['use_gpu']:
            if not torch.cuda.is_available():
                print("There's no GPU is available , Now Automatically converted to CPU device")
            else:
                message = "There's no GPU is available"
                self.device_ids = self.config.GPU['device_id']
                assert len(self.device_ids) > 0,message
                self.device = torch.device('cuda', self.device_ids[0])
                self.use_gpu = True

    def _model_into_cuda(self, model):
        if self.use_gpu:
            model = model.to(self.device)
            if len(self.device_ids) > 1:
                model = nn.DataParallel(model, device_ids = self.device_ids)
    

class EmbeddingLayer(nn.Module):
    def __init__(self, n_d=100, embs=None, fix_emb=True, oov='<oov>', pad='<pad>', normalize=True):
        super(EmbeddingLayer, self).__init__()
        word2id = {}
        if embs is not None:
            embwords, embvecs = embs
            for word in embwords:
                assert word not in word2id, "Duplicate words in pre-trained embeddings"
                word2id[word] = len(word2id)

            sys.stdout.write("{} pre-trained word embeddings loaded.\n".format(len(word2id)))
            # if n_d != len(embvecs[0]):
            #     sys.stdout.write("[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.\n".format(
            #         n_d, len(embvecs[0]), len(embvecs[0])
            #     ))
            n_d = len(embvecs[0])

        # for w in deep_iter(words):
        #     if w not in word2id:
        #         word2id[w] = len(word2id)

        if oov not in word2id:
            word2id[oov] = len(word2id)

        if pad not in word2id:
            word2id[pad] = len(word2id)

        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = nn.Embedding(self.n_V, n_d)
        self.embedding.weight.data.uniform_(-0.25, 0.25)

        if embs is not None:
            weight  = self.embedding.weight
            weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
            sys.stdout.write("embedding shape: {}\n".format(weight.size()))

        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2,1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))

        if fix_emb:
            self.embedding.weight.requires_grad = False

    def forward(self, input):
        return self.embedding(input)



def load_WordLSTM(embedding_path, nclasses, target_model_path, batch_size, config):
    model = WordLSTM(embedding = embedding_path, hidden_size=150, depth=1, dropout=0.3, batch_size = batch_size, nclasses= nclasses, config = config)
    checkpoint = torch.load(target_model_path, map_location='cpu')
    model.load_state_dict(checkpoint)

    device = get_gpu_device(config)
    model.to(device)
    model.eval()
    return model
    

def get_gpu_device(config):
    use_gpu = False
    device_ids = [0]
    device = torch.device('cpu')
    if config.GPU['use_gpu']:
        if not torch.cuda.is_available():
            print("There's no GPU is available , Now Automatically converted to CPU device")
        else:
            message = "There's no GPU is available"
            device_ids = config.GPU['device_id']
            assert len(device_ids) > 0,message
            device = torch.device('cuda', device_ids[0])
            use_gpu = True
    
    return device