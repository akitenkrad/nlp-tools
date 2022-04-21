from pathlib import Path
from typing import Any, Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets.base import BaseDataset
from datasets.msmarco import MsmarcoItemPos, MsmarcoItemX
from matplotlib.colors import cnames
from models.base import BaseModel
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import KFold
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils import Config, Phase, is_notebook
from utils.watchers import LossWatcher

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def bcs_loss(p, y, eps=1e-10):
    '''
    Binary Cross Entropy Loss
    '''
    p = p.type(torch.float64)
    p = torch.clamp(p, min=0.0 + eps, max=1.0 - eps)

    loss = y * torch.log(p) + (1 - y) * torch.log(1 - p)
    loss = torch.sum(loss)
    loss = -loss

    return loss


def evidence_extractor_loss(config: Config, start_prob: torch.Tensor, start_y: torch.Tensor,
                            end_prob: torch.Tensor, end_y: torch.Tensor, psg_ranks: torch.Tensor, psg_ranks_y: torch.Tensor):

    # calculate start loss
    y = torch.zeros_like(start_prob).to(config.train.device)
    y[start_y] = 1.0
    start_loss = bcs_loss(start_prob, y)

    # calculate end_loss
    y = torch.zeros_like(end_prob).to(config.train.device)
    y[end_y] = 1.0
    end_loss = bcs_loss(end_prob, y)

    loss_ap = start_loss.sum() + end_loss.sum()

    # calculate passage ranking loss
    loss_pr = bcs_loss(psg_ranks, psg_ranks_y)

    # calculate final loss
    r = config.evidence_extractor.r
    loss = r * loss_ap + (1 - r) * loss_pr

    return loss


def attention_pooling(v, w1, x1, w2, x2):
    '''
    Args:
        v: nn.Linear(hidden_dim, 1)
        w1, w2: nn.Linear(hidden_dim, hidden_dim)
        x1: [(hidden_dim,)] x m
        x2: (hidden_dim,)
    Return:
        (hidden_dim,)
    '''
    s = torch.stack([v(torch.tanh(w1(x1_j) + w2(x2))) for x1_j in x1]).squeeze()        # (m,)
    a = F.softmax(s.squeeze(), dim=0)                                                   # (m,)
    c_q = torch.stack([a[k] * x1[k] for k in range(len(x1))]).sum(dim=0).squeeze()      # (hidden_dim,)
    return c_q, a


class SentenceEmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding_dim = 300
        self.hidden_dim = 150

        self.char_gru = nn.GRU(self.embedding_dim, self.hidden_dim, 1, batch_first=True, bidirectional=True)
        self.sent_gru = nn.GRU(self.embedding_dim * 2, self.hidden_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, x_words, x_chars):
        '''
        Args:
            x_words_psgs: (sent_len, embedding_dim)
            x_chars_psgs: (sent_len, word_len, embedding_dim)

        Return:
            u: (sent_len, hidden_dim)
            e_words: (sent_len, hidden_dim)
        '''
        h_char = []
        for x_char in x_chars:
            _, _h_char = self.char_gru(x_char.unsqueeze(0))                                     # (1, word_len, embedding_dim) -> (2, 1, hidden_dim)
            h_char.append(_h_char.reshape(1, -1))                                               # (2, 1, hidden_dim) -> (1, embedding_dim)
        h_char = torch.vstack(h_char)                                                           # (1, embedding_dim) x seq_len -> (seq_len, embedding_dim)

        u, _ = self.sent_gru(torch.cat([x_words.unsqueeze(0), h_char.unsqueeze(0)], dim=-1))    # (seq_len, embedding_dim) x 2 -> (1, seq_len, hidden_dim)
        u = u.squeeze()                                                                         # (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)

        return u, x_words                                                                       # (seq_len, hidden_dim, seq_len, embedding_dim)


class EvidenceExtractorLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 300
        self.hidden_dim = 300

        self.v = nn.Parameter(torch.rand((self.embedding_dim, 1), dtype=torch.float32))
        self.v_g = nn.Parameter(torch.rand((self.embedding_dim * 2, 1), dtype=torch.float32))
        self.w_u_q = nn.Parameter(torch.rand((self.embedding_dim, self.embedding_dim), dtype=torch.float32))
        self.w_u_p = nn.Parameter(torch.rand((self.embedding_dim, self.embedding_dim), dtype=torch.float32))
        self.w_v_q = nn.Parameter(torch.rand((self.embedding_dim, self.embedding_dim), dtype=torch.float32))
        self.w_v_p = nn.Parameter(torch.rand((self.embedding_dim, self.embedding_dim), dtype=torch.float32))
        self.w_h_p = nn.Parameter(torch.rand((self.embedding_dim, self.embedding_dim), dtype=torch.float32))
        self.w_h_a = nn.Parameter(torch.rand((self.embedding_dim, self.embedding_dim), dtype=torch.float32))
        self.v_r_q = nn.Parameter(torch.rand((self.embedding_dim, 1), dtype=torch.float32))

        self.w_g = nn.Parameter(torch.rand((self.embedding_dim * 2, self.embedding_dim * 2), dtype=torch.float32))
        self.gru1 = nn.GRU(self.embedding_dim * 2, self.hidden_dim, 1, batch_first=True, bidirectional=False)
        self.gru2 = nn.GRU(self.embedding_dim, self.hidden_dim, 1, batch_first=True, bidirectional=False)

    def forward(self, u_q: Tensor, u_ps: List[Tensor]):
        '''
        'h' is hidden dim = embedding dim
        'm' is the length of a query.
        'n' is the length of a passage.
        'N' is the sum of the lengths of all passages.
        'k' is the number of passages.

        Args:
            u_q (Tensor): query embedding (m x h)
            u_ps (list[Tensor]): passages embedding (n x h) x k

        Return:
            (start, start_probability), (end, end_probability), question_representation
        '''
        v_ps = []
        for u_p in u_ps:        # (n, h)
            _gs = []
            for u_p_t in u_p:   # (h, )

                # Attention Pooling (Rocktaschel et al., 2015)
                _s_t = []
                for u_q_j in u_q:   # (h, )
                    s_j = self.v.T.mm(torch.tanh(self.w_u_q * u_q_j + self.w_u_p * u_p_t))     # (h, )
                    _s_t.append(s_j)
                a_i = torch.softmax(torch.vstack(_s_t), dim=-1)    # (m, h)
                c_q_t = torch.einsum('mh, mh -> h', a_i, u_q)   # (h, )

                # Gated Self-Matching Networks (Wang et al., 2017)
                g = torch.sigmoid(self.w_g.mm(torch.cat([u_p_t, c_q_t], dim=-1).unsqueeze(-1)))  # (2 * h, )
                g = g.squeeze() * torch.cat([u_p_t, c_q_t], dim=-1)
                _gs.append(g)

            gs = torch.vstack(_gs).unsqueeze(0)  # (1, m, 2 * h)
            v_p, _ = self.gru1(gs)
            v_ps.append(v_p.squeeze())

        # Pointer Network
        # 1. predict start point
        _s = []
        for u_q_t in u_q:
            s_t = self.v.T.mm(torch.tanh(self.w_u_q.mm(u_q_t.unsqueeze(-1)) + self.w_v_q.mm(self.v_r_q)))
            _s.append(s_t.squeeze())
        a = torch.softmax(torch.vstack(_s), dim=-1)
        r_q = torch.einsum('m, mh -> h', a.squeeze(), u_q)

        _s = []
        for v_p in v_ps:
            for v_p_t in v_p:
                s_t = self.v.T.mm(torch.tanh(self.w_h_p.mm(v_p_t.unsqueeze(-1)) + self.w_h_a.mm(r_q.unsqueeze(-1))))
                _s.append(s_t)
        a_1 = torch.softmax(torch.vstack(_s), dim=-1)
        p_1 = torch.argmax(a_1.squeeze())

        # 2. predict end point
        _c = []
        for v_p, _a in zip(v_ps, a_1.squeeze()):
            for v_p_t in v_p:
                _c.append(_a * v_p_t)
        c = torch.vstack(_c)
        _, h_t_a = self.gru2(c.unsqueeze(0), r_q.reshape(1, 1, -1))
        h_t_a = h_t_a.reshape(1, 1, -1).squeeze()

        _s = []
        for v_p in v_ps:
            for v_p_t in v_p:
                s_t = self.v.T.mm(torch.tanh(self.w_h_p.mm(v_p_t.unsqueeze(-1)) + self.w_h_a.mm(h_t_a.unsqueeze(-1))))
                _s.append(s_t)
        a_2 = F.softmax(torch.vstack(_s), dim=0)
        p_2 = torch.argmax(a_2)

        # 3. Passage Ranking
        _gs = []
        for v_p in v_ps:
            _s = []
            for v_p_t in v_p:
                s_t = self.v.T.mm(torch.tanh(self.w_v_p.mm(v_p_t.unsqueeze(-1)) + self.w_v_q.mm(r_q.unsqueeze(-1))))
                _s.append(s_t)
            a = torch.softmax(torch.vstack(_s), dim=0)
            r_p = torch.einsum('m, mh -> h', a.squeeze(), v_p)
            g = self.v_g.T.mm(torch.tanh(self.w_g.mm(torch.cat([r_q, r_p]).unsqueeze(-1))))
            _gs.append(g)
        psg_ranks = torch.softmax(torch.stack(_gs).squeeze(), dim=0)

        return (p_1, a_1), (p_2, a_2), psg_ranks


class AnswerSynthesisEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bigru = nn.GRU(hidden_dim, hidden_dim + 1, batch_first=True, bidirectional=True)

    def init_hidden(self, hidden_dim):
        '''initialize hidden state'''
        # TODO add data type
        return torch.zeros(1, 1, hidden_dim).to(device)

    def forward(self, e_q, e_p, f_s, f_e):
        '''
        'm' is the length of a query.
        'n' is the length of a passage.
        'N' is the sum of the lengths of all passages.
        'k' is the number of passages.

        Args:
            e_q: (m, embedding_dim) embeddings of query
            e_p: (N, embedding_dim) embeddings of all passages
            f_s: (N, 1) start point probabilities returned from pointer network
            f_e: (N, 1) end point probabilities returned from pointer network
        '''

        # calculate h_p
        p_len = len(e_p)
        h_p = []
        for i in range(p_len):
            e_p_t, f_s_t, f_e_t = e_p[i], f_s[i], f_e[i]                                # (embedding_dim,), (1,), (1,)
            h_p_t = h_p[-1] if len(h_p) > 0 else self.init_hidden(self.hidden_dim * 2)
            h_p_t = h_p_t[:, :, :self.hidden_dim * 2].reshape(2, 1, -1).sum(dim=0, keepdim=True)

            h_p_t, _ = self.bigru(h_p_t, torch.cat([e_p_t, f_s_t, f_e_t]).reshape(2, 1, -1))

            h_p.append(h_p_t)
        h_p = [h_p_t.squeeze() for h_p_t in h_p]

        # calculate h_q
        q_len = len(e_q)
        h_q = []
        for i in range(q_len):
            e_q_t = e_q[i]                                                              # (embedding_dim,)
            h_q_t = h_q[-1] if len(h_q) > 0 else self.init_hidden(self.hidden_dim * 2)
            h_q_t = h_q_t[:, :, :self.hidden_dim * 2].reshape(2, 1, -1).sum(dim=0, keepdim=True)

            h_q_t, _ = self.bigru(h_q_t, torch.cat([e_q_t, torch.zeros(1), torch.zeros(1)]).reshape(2, 1, -1))

            h_q.append(h_q_t)
        h_q = [h_q_t.squeeze() for h_q_t in h_q]

        h_q = [h_q_t.reshape(2, 1, -1)[:, :, :self.hidden_dim] for h_q_t in h_q]         # (2, 1, hidden_dim)
        h_p = [h_p_t.reshape(2, 1, -1)[:, :, :self.hidden_dim] for h_p_t in h_p]         # (2, 1, hidden_dim)

        h_q = torch.stack(h_q).reshape(-1, self.hidden_dim * 2)
        h_p = torch.stack(h_p).reshape(-1, self.hidden_dim * 2)

        return h_q, h_p


class AnswerSynthesisDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(embedding_dim + hidden_dim * 2, hidden_dim * 2 * 2)
        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.w_d = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=True)
        self.w_a = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.u_a = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.w_r = nn.Linear(embedding_dim, hidden_dim * 2, bias=False)
        self.u_r = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.v_r = nn.Linear(hidden_dim * 2 * 2, hidden_dim * 2, bias=False)
        self.w_o = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, h_q, h_p, ans_embeds):
        '''
        'm' is the length of a query.
        'n' is the length of a passage.
        'N' is the sum of the lengths of all passages.
        'k' is the number of passages.

        Args:
            h_q: (m, hidden_dim)
            h_p: (N, hidden_dim)
            ans_embeds: (ans_len, embedding_dim)
        '''
        ans_len = len(ans_embeds)
        h = torch.cat([h_q, h_p], dim=0).sum(dim=0)
        d = None
        probs = []
        for i in range(ans_len):
            w = ans_embeds[i]                                                               # (embedding_dim, )
            if d is None:
                d = torch.tanh(self.w_d(torch.stack([h_q[0], h_p[0]])))                     # (hidden_dim * 2, )
            else:
                d = d.reshape(2, -1)
            c, _ = attention_pooling(self.v, self.w_a, d, self.u_a, h)                      # (hidden_dim * 2, )

            _, d = self.gru(torch.stack([w, c]).reshape(1, 1, -1), d.reshape(1, 1, -1))

            r_t = self.w_r(w) + self.u_r(c) + self.v_r(d.squeeze())
            r_t = r_t.reshape(2, -1).max(dim=0).values
            prob = F.softmax(self.w_o(r_t), dim=0)

            probs.append(prob)

        return probs


class SNetEvidenceExtractor(BaseModel):
    def __init__(self, config: Config, name='snet-evidence-extractor'):
        super().__init__(config, name)

        self.build()

    def build(self):
        self.sentence_embedding = SentenceEmbeddingLayer()
        self.evidence_extractor = EvidenceExtractorLayer()

    def step(self, x: MsmarcoItemX, y: MsmarcoItemPos, loss_func: Callable) -> Tuple[Any, Any]:

        # sentence embedding
        query_word_tokens = x.query_word_tokens.to(self.config.train.device)
        query_char_tokens = [char.to(self.config.train.device) for char in x.query_char_tokens]

        u_q, uq_wdembd = self.sentence_embedding(query_word_tokens, query_char_tokens)
        u_ps, up_wdembds = [], []
        for pw_tokens, pc_tokens in zip(x.passage_word_tokens, x.passage_char_tokens):
            pw_tokens = pw_tokens.to(self.config.train.device)
            pc_tokens = [char.to(self.config.train.device) for char in pc_tokens]
            u_p, up_wdembd = self.sentence_embedding(pw_tokens, pc_tokens)
            u_ps.append(u_p)
            up_wdembds.append(up_wdembd)

        # evidence_extractor
        (p_1, a_1), (p_2, a_2), psg_ranks = self.evidence_extractor(u_q, u_ps)

        loss = loss_func(self.config, p_1, y.start_pos.to(self.config.train.device), p_2, y.end_pos.to(self.config.train.device), psg_ranks, x.passage_is_selected.to(self.config.train.device))

        return loss, ((p_1, a_1), (p_2, a_2), psg_ranks)

    def step_wo_loss(self, x: MsmarcoItemX, y: MsmarcoItemPos) -> Any:
        # sentence embedding
        u_q, uq_wdembd = self.sentence_embedding(x.query_word_tokens, x.query_char_tokens)
        u_ps, up_wdembds = [], []
        for pw_tokens, pc_tokens in zip(x.psg_word_tokens, x.psg_char_tokens):
            u_p, up_wdembd = self.sentence_embedding(pw_tokens, pc_tokens)
            u_ps.append(u_p)
            up_wdembds.append(up_wdembd)

        # evidence_extractor
        (p_1, a_1), (p_2, a_2), psg_ranks = self.evidence_extractor(u_q, u_ps)

        return ((p_1, a_1), (p_2, a_2), psg_ranks)

    def validate(self, epoch: int, valid_dl: DataLoader, loss_func: Callable):
        loss_watcher = LossWatcher('loss')
        with tqdm(valid_dl, total=len(valid_dl), desc=f'[Epoch {epoch:4d} - Validate]', leave=False) as valid_it:
            for x, y in valid_it:
                with torch.no_grad():
                    loss, out = self.step(x, y, loss_func)
                    loss_watcher.put(loss.item())
        return loss_watcher.mean

    def forward(self, x):
        '''execute forward process in the PyTorch module'''
        pass

    def fit(self, ds: BaseDataset, optimizer: optim.Optimizer, lr_scheduler: Any, loss_func: Callable):
        '''train the model

        Args:
            ds (BaseDataset): dataset.
            optimizer (optim.Optimizer): optimizer.
            lr_scheduler (Any): scheduler for learning rate. ex. optim.lr_scheduler.ExponentialLR.
            loss_func (Callable): loss function
        '''
        self.train().to(self.config.train.device)
        ds.to_train()
        global_step = 0
        tensorboard_dir = self.config.log.log_dir / 'tensorboard' / f'exp_{self.config.now().strftime("%Y%m%d-%H%M%S")}'

        with SummaryWriter(str(tensorboard_dir)) as tb_writer:
            kfold = KFold(n_splits=self.config.train.k_folds, shuffle=True)
            with tqdm(enumerate(kfold.split(ds)), total=self.config.train.k_folds, desc=f'[Fold {0:2d}]') as fold_it:
                for fold, (train_indices, valid_indices) in fold_it:
                    fold_it.set_description(f'[Fold {fold:2d}]')

                    # prepare dataloader
                    train_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)
                    valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_indices)
                    train_dl = DataLoader(ds, batch_size=self.config.train.batch_size, sampler=train_subsampler, collate_fn=ds.collate_fn)
                    valid_dl = DataLoader(ds, batch_size=self.config.train.batch_size, sampler=valid_subsampler, collate_fn=ds.collate_fn)

                    # initialize learning rate
                    optimizer.param_groups[0]['lr'] = self.config.train.lr

                    # Epoch roop
                    with tqdm(range(self.config.train.epochs), total=self.config.train.epochs, desc=f'[Fold {fold:2d} | Epoch   0]', leave=False) as epoch_it:
                        valid_loss_watcher = LossWatcher('valid_loss', patience=self.config.train.early_stop_patience)
                        for epoch in epoch_it:
                            loss_watcher = LossWatcher('loss')

                            # Batch roop
                            with tqdm(enumerate(train_dl), total=len(train_dl), desc=f'[Epoch {epoch:3d} | Batch {0:3d}]', leave=False) as batch_it:
                                for batch, (x, y) in batch_it:

                                    # process model and calculate loss
                                    loss_at_batch = []
                                    for _x, _y in zip(x, y):
                                        _loss, out = self.step(_x, _y, loss_func)
                                        loss_at_batch.append(_loss)
                                    loss = torch.stack(loss_at_batch).mean()

                                    # update parameters
                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()

                                    # put logs
                                    tb_writer.add_scalar('train loss', loss.item(), global_step)
                                    loss_watcher.put(loss.item())
                                    batch_it.set_description(f'[Epoch {epoch:3d} | Batch {batch:3d}] Loss: {loss.item():.3f}')

                                    # global step for summary writer
                                    global_step += 1

                                    if batch > 0 and batch % self.config.train.logging_per_batch == 0:
                                        log = f'[Fold {fold:02d} | Epoch {epoch:03d} | Batch {batch:05d}/{len(train_dl):05d} ({(batch/len(train_dl)) * 100.0:.2f}%)] Loss:{loss.item():.3f}'
                                        self.config.log.train.info(log)

                            # end of Batch
                            # evaluation
                            val_loss = self.validate(epoch, valid_dl, loss_func)

                            # step scheduler
                            lr_scheduler.step()

                            # update iteration description
                            desc = f'[Fold {fold:2d} | Epoch {epoch:3d}] Train Loss: {loss_watcher.mean:.5f} | Valid Loss:{val_loss:.5f}'
                            epoch_it.set_description(desc)

                            # logging
                            last_lr = lr_scheduler.get_last_lr()[0]
                            log = f'[Fold {fold:2d} / Epoch {epoch:3d}] Train Loss: {loss_watcher.mean:.5f} | Valid Loss:{val_loss:.5f} | LR: {last_lr:.7f}'
                            self.config.log.train.info(log)
                            tb_writer.add_scalar('valid loss', val_loss, global_step)

                            # save best model
                            if valid_loss_watcher.is_best:
                                self.save_model(f'{self.name}_best.pt')
                            valid_loss_watcher.put(val_loss)

                            # save model regularly
                            if epoch % 5 == 0:
                                self.save_model(f'{self.name}_f{fold}e{epoch}.pt')

                            # early stopping
                            if valid_loss_watcher.early_stop:
                                self.config.log.train.info(f'====== Early Stopping @epoch: {epoch} @Loss: {valid_loss_watcher.best_score:5.10f} ======')
                                break

                    # end of Epoch
                    # backup files
                    if self.config.backup.backup:
                        self.config.log.train.info('start backup process')
                        self.config.backup_logs()
                        self.config.log.train.info(f'finished backup process: backup logs -> {str(Path(self.config.backup.backup_dir).resolve().absolute())}')

                    self.save_model(f'{self.name}_last_f{fold}.pt')

            # end of k-fold
            self.config.backup_logs()

    def find_lr(self, ds: BaseDataset, optimizer: optim.Optimizer, loss_func: Callable, batch_size=8, init_value=1e-8, final_value=10.0, beta=0.98):
        self.train().to(self.config.train.device)
        self.config.add_logger('lr_finder', silent=True)
        ds.to_train()
        dl = DataLoader(ds, batch_size=batch_size)
        num = len(dl) - 1
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.0
        best_loss = 0.0
        losses = []
        log_lrs = []
        stop_counter = 10

        with tqdm(enumerate(dl), total=len(dl), desc='[B:{:05d}] lr:{:.8f} best_loss:{:.3f}'.format(0, lr, -1)) as it:
            for idx, (x, y) in it:

                # process model and calculate loss
                loss, out = self.step(x, y, loss_func)

                # compute the smoothed loss
                avg_loss = beta * avg_loss + (1 - beta) * loss.item()
                smoothed_loss = avg_loss / (1 - beta**(idx + 1))

                # stop if the loss is exploding
                if idx > 0 and smoothed_loss > 1e+3 * (best_loss + 1e-10):
                    stop_counter -= 1

                    if stop_counter <= 0:
                        break

                # record the best loss
                if smoothed_loss < best_loss or idx == 0:
                    best_loss = smoothed_loss

                # store the values
                losses.append(smoothed_loss)
                log_lrs.append(lr)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update progress bar
                desc = '[B:{:05d}] lr:{:.10f} best_loss:{:.6f} loss:{:.6f}'.format(idx + 1, lr, best_loss, loss.item())
                it.set_description(desc)
                self.config.log.lr_finder.info(desc)

                # update learning rate
                lr *= mult
                optimizer.param_groups[0]['lr'] = lr

            # save figure
            save_path = self.config.log.log_dir / 'lr_finder' / f'{self.name}_lr_loss_curve.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(16, 8))
            plt.plot(log_lrs[10:-5], losses[10:-5])
            plt.xscale('log')
            plt.xlabel('Learning Rate', fontsize=16)
            plt.ylabel('Loss', fontsize=16)
            plt.savefig(str(save_path))
            plt.close()
            self.config.log.lr_finder.info(f'saved -> {str(save_path.resolve().absolute())}')

    def predict(self, ds: BaseDataset, phase: Phase) -> Tuple[np.ndarray, np.ndarray]:
        '''predict

        Args:
            ds (BaseDataset): instance of BaseDataset
            phase (Phase): one of phases

        Returns:
            outputs, labels
        '''
        self.eval().to(self.config.train.device)
        ds.to(phase)
        dl = DataLoader(ds, batch_size=self.config.train.batch_size)

        results = []
        labels = []
        for x, y in tqdm(dl, total=len(dl)):
            with torch.no_grad():
                out = self.step_wo_loss(x, y)
            results.append(out.cpu().numpy().copy())
            labels.append(y.cpu().numpy().copy())

        results_res = np.concatenate(results, axis=0)
        labels_res = np.concatenate(labels, axis=0)
        return results_res, labels_res

    def describe(self, ds: BaseDataset):
        x, y = next(iter(DataLoader(ds, batch_size=1)))
        self.config.describe_model(self, input_size=x.shape)

    def evaluate_binary_problem(self, ds: BaseDataset):
        '''evaluate binary problem

        Args:
            ds (BaseDataset): dataset

        Returns:
            metrics dict. (auc, accuracy, precision, recall, f1)
        '''
        scores, labels = self.predict(ds, Phase.TEST)
        preds = [1 if s > 0.5 else 0 for s in scores]

        auc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)

        self.config.log.logger.info(f'{self.name} - auc:       {auc:0.3f}')
        self.config.log.logger.info(f'{self.name} - accuracy:  {accuracy:0.3f}')
        self.config.log.logger.info(f'{self.name} - precision: {precision:0.3f}')
        self.config.log.logger.info(f'{self.name} - recall:    {recall:0.3f}')
        self.config.log.logger.info(f'{self.name} - F1:        {f1:0.3f}')

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color=cnames['salmon'], lw=2, label=f'ROC Curve (AREA = {auc:0.2f})')
        plt.plot([0, 1], [0, 1], color=cnames['dodgerblue'], lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('ROC CURVE', fontsize=16)
        plt.legend(loc='lower right', fontsize=15)
        plt.savefig(str(self.config.log.log_dir / f'{self.name}_roc_curve.png'))
        plt.close()

        return {
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
