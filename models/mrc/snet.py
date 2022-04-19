from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Vocabulary
from torch import Tensor
from utils.glove import load_glove_weights_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class SentenceEmbedding(nn.Module):
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

        u, _ = self.sent_gru(torch.cat([e_words.unsqueeze(0), h_char.unsqueeze(0)], dim=-1))    # (seq_len, embedding_dim) x 2 -> (1, seq_len, hidden_dim)
        u = u.squeeze()                                                                         # (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)

        return u, e_words                                                                       # (seq_len, hidden_dim, seq_len, embedding_dim)


class EvidenceExtractor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.v = nn.Linear(embedding_dim, 1, bias=False)
        self.v_g = nn.Linear(embedding_dim * 2, 1, bias=False)
        self.w_u_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_u_p = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_v_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_v_p = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_h_p = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_h_a = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_r_q = nn.Parameter(torch.zeros((self.embedding_dim,), dtype=torch.float32))

        self.w_g = nn.Linear(embedding_dim * 2, embedding_dim * 2, bias=False)
        self.gru1 = nn.GRU(embedding_dim * 2, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(embedding_dim, hidden_dim, 1, batch_first=True, bidirectional=True)

    def init_hidden(self, hidden_dim):
        '''initialize hidden state'''
        # TODO add data type
        return torch.zeros(1, 1, hidden_dim).to(device)

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
        for u_p in u_ps:

            # Attention Pooling (Rocktaschel et al., 2015)
            s = self.v(torch.tanh(self.w_u_q(u_q) + self.w_u_p(u_p)))
            a = torch.softmax(s, dim=-1)
            c_q = a * u_q

            # Gated Self-Matching Networks (Wang et al., 2017)
            g = torch.sigmoid(self.w_g(torch.cat([u_p, c_q], dim=-1)))
            g = g * torch.cat([u_p, c_q], dim=-1)
            v_p, _ = self.gru1(g.unsqueeze(0))

            v_ps.append(v_p)

        # Pointer Network
        N = len(v_ps)
        # 1. predict start point

        # initial hidden state "r_q"
        s = self.v(torch.tanh(self.w_u_q(u_q) + self.w_v_q(self.v_r_q)))
        a = torch.softmax(s, dim=-1)
        r_q = torch.einsum('m, mh -> h', a.squeeze(), u_q)

        s = self.v(torch.tanh(self.w_h_p(torch.cat(v_ps, dim=0)) + self.w_h_a(r_q)))
        a_1 = torch.softmax(s, dim=-1)
        p_1 = torch.argmax(a_1.squeeze())

        # calculate start point

        #   2. predict end point
        c = torch.einsum('ij, kij -> j', a, torch.cat(v_ps, dim=0))
        _, h_t_a = self.gru2(c.reshape(1, 1, -1), r_q.reshape(2, 1, -1))
        h_t_a = h_t_a.reshape(1, 1, -1).squeeze()

        s = self.v(torch.tanh(self.w_h_p(torch.cat(v_ps, dim=0)) + self.w_h_a(h_t_a)))
        a_2 = F.softmax(s.squeeze(), dim=0)
        p_2 = torch.argmax(a_2)

        # Passage Ranking
        gs = []
        for v_p in v_ps:
            s = self.v(torch.tanh(self.w_v_p(v_p) + self.w_v_q(r_q)))
            a = torch.softmax(s.squeeze(), dim=0)
            r_p = torch.einsum('m, mh -> h', a, v_p.squeeze())
            g = self.v_g(torch.tanh(self.w_g(torch.cat([r_q, r_p]))))
            gs.append(g)

        psg_ranks = torch.softmax(torch.stack(gs).squeeze(), dim=0)

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