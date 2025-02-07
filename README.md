# Learning from Mistakes: Memory-Based Continual Learning

A simple approach to continual learning where the model only learns from its mistakes through a memory buffer.

## Key Idea
Instead of training directly on tasks, we:
1. Do inference on current examples
2. Store mistakes in a memory buffer
3. Train ("dream") on the memory until we learn those examples well
4. Repeat

This helps prevent catastrophic forgetting without complex mechanisms.

## Results on MNIST
We tested on pairs of visually similar digits:
1. First pair (1 vs 7): Learn to 90% accuracy
2. Second pair (3 vs 8): Learn to 90% while maintaining ~86% on previous
3. Third pair (4 vs 9): Learn to 90% while maintaining ~77-94% on previous pairs

Compare this to regular training which completely forgets previous pairs (0% accuracy).

## Implementation
The core code is in `experiments/scripts/train_mnist.py` and includes:
- Simple CNN for MNIST
- Memory buffer for storing mistakes
- Dream phase that trains on memory until 97% accuracy
- Comparison with baseline that trains normally

## Running the Code
```bash
# Install dependencies
uv venv
uv pip install -r requirements.txt

# Run experiment
PYTHONPATH=$PYTHONPATH:. uv run experiments/scripts/train_mnist.py
```

## Next Steps
1. Test on more complex datasets (CIFAR, ImageNet)
2. Compare with other continual learning methods
3. Analyze what makes some examples harder to remember

## License
[License information to be added] 