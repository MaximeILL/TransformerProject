import torch
import torch.nn as nn
import torch.optim as optim

from models.transformer import Transformer
from optimizers.ademamix import AdEMAMix
from data.dataset import create_dataset, TDataset
from utils.visualization import plot_loss_curves, plot_attention_weights

def train_model(optimizer_name='Adam', num_epochs=50):
    # model params
    src_vocab_size = 50
    tgt_vocab_size = 50
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 256
    max_seq_length = 10
    dropout = 0.1

    # init
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(transformer.parameters(), lr=0.001)
    elif optimizer_name == 'AdEMAMix':
        optimizer = AdEMAMix(transformer.parameters(), lr=0.001)
    else:
        raise ValueError("not supported optimizer")

    # dataset creation
    num_samples = 1000
    seq_length = 10
    vocab_size = 50
    src_data, tgt_data = create_dataset(num_samples, seq_length, vocab_size)
    dataset = TDataset(src_data, tgt_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # training loop
    losses = []
    transformer.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for src_batch, tgt_batch in dataloader:
            optimizer.zero_grad()
            src_batch = src_batch
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            output = transformer(src_batch, tgt_input)
            loss = criterion(output.view(-1, tgt_vocab_size), tgt_output.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"{optimizer_name} - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return losses, transformer, src_data, tgt_data

def main():
    num_epochs = 50
    # train and visualize for Adam
    adam_losses, adam_transformer, src_data, tgt_data = train_model(optimizer_name='Adam', num_epochs=num_epochs)
    # train and visualize for AdEMAMix
    ademamix_losses, ademamix_transformer, _, _ = train_model(optimizer_name='AdEMAMix', num_epochs=num_epochs)

    plot_loss_curves(adam_losses, ademamix_losses, num_epochs)

    src_example = src_data[0].unsqueeze(0)
    tgt_example = tgt_data[0].unsqueeze(0)

    # Adam attention weights 
    adam_transformer.eval()
    with torch.no_grad():
        output = adam_transformer(src_example, tgt_example[:, :-1])
    adam_encoder_attn = adam_transformer.encoder_attn_weights
    adam_decoder_self_attn = adam_transformer.decoder_self_attn_weights
    adam_decoder_cross_attn = adam_transformer.decoder_cross_attn_weights

    # AdEMAMix attention weights
    ademamix_transformer.eval()
    with torch.no_grad():
        output = ademamix_transformer(src_example, tgt_example[:, :-1])
    ademamix_encoder_attn = ademamix_transformer.encoder_attn_weights
    ademamix_decoder_self_attn = ademamix_transformer.decoder_self_attn_weights
    ademamix_decoder_cross_attn = ademamix_transformer.decoder_cross_attn_weights

    src_tokens = [str(tok.item()) for tok in src_example[0]]
    tgt_tokens = [str(tok.item()) for tok in tgt_example[0][:-1]]

    # plot attention weights

    plot_attention_weights(
        adam_encoder_attn,
        ademamix_encoder_attn,
        layer_num=0,
        head_num=0,
        src_tokens=src_tokens,
        tgt_tokens=src_tokens,
        attention_type='Encoder Self-Attention'
    )
    plot_attention_weights(
        adam_decoder_self_attn,
        ademamix_decoder_self_attn,
        layer_num=0,
        head_num=0,
        src_tokens=tgt_tokens,
        tgt_tokens=tgt_tokens,
        attention_type='Decoder Self-Attention'
    )
    plot_attention_weights(
        adam_decoder_cross_attn,
        ademamix_decoder_cross_attn,
        layer_num=0,
        head_num=0,
        src_tokens=src_tokens,
        tgt_tokens=tgt_tokens,
        attention_type='Decoder Cross-Attention'
    )

if __name__ == "__main__":
    main()
