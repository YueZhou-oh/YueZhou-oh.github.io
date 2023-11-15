# transformer-xl

> - memory与上一层的输出，在sequence length维度concatenation，作为生成KV的输入，KV对应的weight不变，依然为[h, h]；经过self-attention后KV的sequence length维度被抵消，因此推理时，可以把多个segment的memory一起输入给KV，因此推理时可以看到很长的sequence；
> - memory不作用与query，因此生成Q的输入以及weight与原始transformer一致

``` python
    # init():
    self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
    self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

    # forward():
    if mems is not None:
        c = torch.cat([mems, h], 0)
    else:
        c = h

    head_q = self.q_net(h)                                  # q without memory
    head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)     # k and v include memory

    head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)    # [qlen x bsz x n_head x d_head]
    head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)    # [klen x bsz x n_head x d_head]
    head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)    # [klen x bsz x n_head x d_head]

    # [qlen x klen x bsz x n_head]          # eg. klen = 2 x qlen
    attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
    attn_score.mul_(self.scale)

    # [qlen x klen x bsz x n_head]
    attn_prob = F.softmax(attn_score, dim=1)
    attn_prob = self.dropatt(attn_prob)

    # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
    attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
    attn_vec = attn_vec.contiguous().view(
        attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

```

# RMT
> - 在transformer-xl的基础上，在Q的维度，添加了[mem] token，生成Q的输入为[read, h, write]的concatenation，因此RMT的memory is effectively deeper，同时训练所需显存与计算量也更大
> - 因为memory token的参数也要更新，因此训练过程类似BPTT的概念

``` python
    # forward():
    word_emb = self.word_emb(dec_inp)

    mlen = mems[0].size(0) if mems is not None else 0
    
    # Concat with mem_tokens
    if mem_tokens is not None:
        word_emb = torch.cat((mem_tokens, word_emb), dim=0)
        if self.mem_at_end:
            word_emb = torch.cat((word_emb, mem_tokens), dim=0)

    # qlen, bsz = dec_inp.size()
    qlen = word_emb.shape[0]
    klen = mlen + qlen

```
# megabyte
不是memory transformer；
分两个阶段做next word & next byte prediction：
1. 降低计算复杂度，允许更长序列的输入；
2. 虽然在NLP上只能达到SOTA相当的结果,但对于长序列（信号、RNA等）可能有参考价值；