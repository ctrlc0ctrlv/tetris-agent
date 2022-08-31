# tetris-agent
Neural network RL agent for tetris OpenAI gym environment

## ~~First implementation (fully-connected layers)~~

- Linear(state, 100)
- ReLU
- Linear(100, 50)
- ReLU
- Linear(50, action)

## Second implementation (better performance)

- Convolutional(in_channels=1, out_channels=5, kernel_size=3)
- ReLU
- MaxPooling(kernel_size=2)
- Flattening
- Linear(80, 128)
- ReLU
- Linear(128, n_actions)