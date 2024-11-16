# Transformer Project

This project implements a Transformer model from scratch in PyTorch, trained with the Adam and AdEMAMix optimizers.
It compares the performance of both optimizers by plotting the loss curves and visualizing the attention weights (Encoder Self-Attention, Decoder Self-Attention, Decoder Cross-Attention).

## Description

- **models/transformer.py**: Contains the implementation of the Transformer model.
- **optimizers/ademamix.py**: Implements the AdEMAMix optimizer.
- **data/dataset.py**: Manages the creation of the simulated dataset.
- **utils/visualization.py**: Contains functions to plot the loss curves and attention weight heatmaps.
- **main.py**: Main script for training and visualization.

## Installation

* Clone the repository :

```bash
git clone https://github.com/MaximeILL/TransformerProject.git
```

* Navigate to the directory :
  
```bash
cd TransformerProject
```

* Install the dependencies :
  
```bash
pip install -r requirements.txt
```

## Run

* Ensure you have installed the dependencies as described in the Installation section.

* Execute the main script :

```bash
python main.py
```

## Ressources

* [PyTorch Documentation](https://pytorch.org/)
* [AdEMAMix paper](https://arxiv.org/abs/2409.03137)
