# QPCNet: QPCNet: A Hybrid Quantum Positional Encoding and Channel Attention Network for Image Classification

This repository contains the code and data used in the QPCNet project.

## Dataset Information

This project uses the following datasets: MNIST, FashionMNIST, and CIFAR-10. These datasets will be automatically downloaded during the execution of the training scripts. If automatic download fails due to network restrictions or other issues, you can manually download the datasets from the official websites:

- **MNIST**: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- **FashionMNIST**: [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- **CIFAR-10**: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

After downloading, place the datasets in the `data/` folder according to the training script requirements.

## Installation

The project requires Python 3.7+ and the following dependencies:

```bash
pip install torch torchvision
pip install pennylane
```

Additional dependencies can be installed as needed.

## Usage and Code Maintenance Notice

To run the project, navigate to the `src/` directory and execute the corresponding training script:

```bash
python train_{dataset_name}.py
```

Running the training script will automatically generate the required result figures for analysis.

Please note that some parts of `QPCNet.py` are currently under maintenance:

```python
# QPE implementation is under maintenance and will be publicly released soon.
# QCA implementation is under maintenance and will be publicly released soon.
```

Our team will upload the complete and finalized code in the near future.

