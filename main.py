#!/usr/bin/env python3

import torch
import torch.nn as nn

from model import Model, create_mask, generate_square_subsequent_mask
from data import Data

PERM_LENGTH = 7
D_EMB = 4 
BATCH_SIZE = 256 
NUM_EPOCHS = 30 

transformer = Model(d_emb=D_EMB)

# init parameters
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

data = Data('./data')

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)


def train(model, optimizer):
    model.train()
    total_loss = 0.0

    train_dataloader = data.train_dataloader(BATCH_SIZE)

    for src, tgt in train_dataloader:

        src_mask, tgt_mask = create_mask()
        logits = model(src, tgt, src_mask, tgt_mask)

        optimizer.zero_grad()

        loss = loss_fn(logits, tgt)
        
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    total_loss = 0.0

    validate_dataloader = data.validate_dataloader(BATCH_SIZE)

    for src, tgt in validate_dataloader:

        src_mask, tgt_mask = create_mask()
        logits = model(src, tgt, src_mask, tgt_mask)

        loss = loss_fn(logits, tgt)

        total_loss += loss.item()
    
    return total_loss / len(list(validate_dataloader))


from timeit import default_timer as timer

best_val_loss = None
for epoch in range(0, NUM_EPOCHS):
    start_time = timer()
    train_loss = train(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)

    print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")


def greedy_decode(model, src):
    model.eval()

    src_mask, _ = create_mask()
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for _ in range(PERM_LENGTH):
        tgt_mask = generate_square_subsequent_mask(ys.size(1))
        out = model.decode(ys, memory, tgt_mask)

        logits = model.generator(out[:, -1])
        _, next_word = torch.max(logits, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1,1).type_as(src.data).fill_(next_word)], dim=1
        )

    return ys[0]





test_cycle = data.line2tensor("3 5 2")
print(test_cycle)

permutation = greedy_decode(transformer, test_cycle)
print(permutation)
