import matplotlib.pyplot as plt
import seaborn as sns

# plot loss curves
def plot_loss_curves(adam_losses, ademamix_losses, num_epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), adam_losses, marker='o', linestyle='-', label='Adam')
    plt.plot(range(1, num_epochs + 1), ademamix_losses, marker='s', linestyle='--', label='AdEMAMix')
    plt.title('Loss for Adam and AdEMAMix')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

# plot attention weights
def plot_attention_weights(attention_weights1, attention_weights2, layer_num, head_num, src_tokens, tgt_tokens, attention_type='Encoder Self-Attention'):
    attn1 = attention_weights1[layer_num][0, head_num].cpu().detach().numpy()
    attn2 = attention_weights2[layer_num][0, head_num].cpu().detach().numpy()
    attn1 = attn1[:len(tgt_tokens), :len(src_tokens)]
    attn2 = attn2[:len(tgt_tokens), :len(src_tokens)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(attn1, xticklabels=src_tokens, yticklabels=tgt_tokens, cmap='viridis', ax=axes[0])
    axes[0].set_xlabel('Source tokens')
    axes[0].set_ylabel('Target tokens')
    axes[0].set_title(f'Adam - {attention_type} - Layer {layer_num+1}, Head {head_num+1}')

    sns.heatmap(attn2, xticklabels=src_tokens, yticklabels=tgt_tokens, cmap='viridis', ax=axes[1])
    axes[1].set_xlabel('Source tokens')
    axes[1].set_ylabel('Target tokens')
    axes[1].set_title(f'AdEMAMix - {attention_type} - Layer {layer_num+1}, Head {head_num+1}')

    plt.tight_layout()
    plt.show()
