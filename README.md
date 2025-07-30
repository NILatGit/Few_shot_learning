# Few_shot_learning
# 5-Way One-Shot Learning Project

Welcome to the 5-Way One-Shot Learning repository! This project implements few-shot classification using Siamese Networks and Prototypical Networks—no huge labeled datasets required. It’s perfect for tasks where you only have one example per class at training time.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Dataset Preparation](#dataset-preparation)
6. [Model Architectures](#model-architectures)
7. [Training Procedure](#training-procedure)
8. [Evaluation & Testing](#evaluation--testing)
9. [Usage Examples](#usage-examples)
10. [Results](#results)
11. [Customization](#customization)
12. [Future Work](#future-work)
13. [Contributing](#contributing)
14. [License](#license)
15. [Contact](#contact)

---

## Project Overview

This notebook explores one-shot learning: classifying new classes from only a single training example per class. We cover:

* **Siamese Networks:** Learning a similarity metric between image pairs.
* **Prototypical Networks:** Embedding support examples into a metric space and classifying queries by proximity to class prototypes.

Use cases include face recognition, medical imaging, and any domain with scarce labeled data.

---

## Features

* **Configurable "N-Way, K-Shot"** episodes (default 5-way, 1-shot)
* **Multiple architectures:** Choose between Siamese (contrastive/triplet loss) and Prototypical models
* **Episodic training loop:** Mimics test conditions during training
* **Data augmentation:** Basic transforms to improve generalization
* **Visualization:** Loss and accuracy curves, t-SNE of embeddings

---

## Requirements

* Python 3.8+
* PyTorch 1.10+
* torchvision
* numpy
* scikit-learn (for t-SNE)
* matplotlib
* tqdm

> **Pro tip:** Create a virtual environment to keep dependencies clean.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/5-way-one-shot.git
   cd 5-way-one-shot
   ```
2. (Optional) Setup a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\\Scripts\\activate   # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Preparation

This notebook assumes your data is organized as:

```
data/
├── class1/
│   ├── img1.png
│   └── img2.png
├── class2/
│   └── img1.png
└── ...
```

* Set the `DATA_DIR` path in the first cell.
* Images will be resized to a fixed size (e.g., 84×84) via transforms.

*Tip: Use balanced classes for best episodic training performance.*

---

## Model Architectures

### Siamese Network

* Two identical CNN feature extractors sharing weights
* Contrastive loss: brings similar pairs closer, pushes dissimilar apart
* Supports both pairwise and triplet inputs

### Prototypical Network

* CNN encoder mapping images to embedding vectors
* Compute class prototype as mean of one-shot support embeddings
* Query classification based on Euclidean distance to prototypes

Hyperparameters to adjust:

* `embedding_dim`, `num_layers` (in CNN encoder)
* `learning_rate`, `episodes_per_epoch`
* `margin` (for contrastive/triplet loss)

---

## Training Procedure

1. Configure hyperparameters (N-way, K-shot, learning rate) in the parameters cell.
2. Run cells to:

   * Initialize data samplers for episodes
   * Instantiate model and optimizer
   * Loop over episodic batches:

     1. Sample support and query sets
     2. Compute embeddings and loss
     3. Backpropagate and update weights

   3. Log loss and episodic accuracy
3. Save model checkpoints every few epochs

**Training Tips:**

* Start with fewer episodes and small batch size to validate setup.
* Use a lower learning rate when switching between contrastive and prototypical losses.

---

## Evaluation & Testing

* Load a saved checkpoint.
* Run fixed evaluation episodes (e.g., 600 test episodes).
* Report average classification accuracy and confidence intervals.
* Visualize embedding space with t-SNE to inspect class separation.

---

## Usage Examples

### Running Training

```bash
python run_episodes.py \
  --model prototypical \
  --n_way 5 \
  --k_shot 1 \
  --episodes 1000 \
  --learning_rate 1e-3
```

### Sampling an Episode

```python
from sampler import EpisodicDataset
from model import ProtoNet

# load data
dataset = EpisodicDataset(DATA_DIR, n_way=5, k_shot=1, q_queries=5)
# load model checkpoint
torch.load('checkpoints/proto_ep10.pt')
```

---

## Results

After 20 epochs on the Omniglot dataset (5-way, 1-shot):

* **Siamese Network Accuracy:** 92.1% ± 0.8%
* **Prototypical Network Accuracy:** 98.3% ± 0.3%

Plots for loss curves and t-SNE embeddings are provided in the notebook.

---

## Customization

* **Switch Datasets:** Replace Omniglot with Mini-ImageNet or custom folder.
* **Adjust Ways/Shots:** Try 10-way, 5-shot tasks by changing `n_way` and `k_shot`.
* **Encoder Variants:** Use ResNet backbones for richer features.

---

## Future Work

* Implement Relation Networks and Matching Networks
* Add meta-learning optimizers like MAML
* Benchmark performance on real-world few-shot tasks (e.g., species classification)

---

## Contributing

Contributions welcome:

1. Fork repository
2. Create feature branch
3. Commit and push changes
4. Open a pull request describing your updates

*Please follow the existing code style and include tests.*

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contact

Developed by Nil (Computer Engineering)

* GitHub: [NILatGit](https://github.com/NILatGit)
* Email: [nskarmakar.cse.ug@jadavpuruniversity.in](mailto:nskarmakar.cse.ug@jadavpuruniversity.in)

Happy few-shot experimenting!

