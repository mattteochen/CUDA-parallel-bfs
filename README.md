# CUDA Parallel Breadth First Search Algorithm Implementation

This project focuses on implementing the Breadth First Search (BFS) algorithm using CUDA, a parallel computing platform, to achieve significant performance improvements compared to a simple sequential implementation. The primary goal is to explore the potential of GPU acceleration for graph traversal tasks.

For more information on the approach and its benefits, you can refer to the Nvidia paper titled [Scalable GPU Graph Traversal](https://research.nvidia.com/publication/scalable-gpu-graph-traversal) authored by D. Merill, M. Garland, and A. Grimshaw.

## How to Run

To compile and run the code, you need to set the `USE_HOST` compilation flag to specify whether you want to run the algorithm sequentially (set to `1`) or in parallel on a CUDA-enabled GPU (set to `0`).

Here are the steps to run the code:

1. Open the [Google Colab notebook](https://colab.research.google.com/drive/1D8x4Yx-GcRYlByXMa87IGPMnLZn7LvPZ?usp=sharing) provided in this repository and initialize the environment.

2. Load the desired test cases from the `test` folder or create your own test cases if needed.

3. Run the CUDA program to execute the BFS algorithm.

Please note that a CUDA-enabled GPU is required to run the parallel version of the algorithm effectively.

## Key Findings

The parallel implementation of the BFS algorithm outperforms the sequential version in a noticeable manner. One particularly interesting aspect is the performance improvement achieved by leveraging the device's shared block memory for writing the neighborhood nodes (`next_queue`) instead of directly using global memory, as commonly described in the literature.

## Credits

- [D. Merill, M. Garland, A. Grimshaw - "Scalable GPU Graph Traversal"](https://research.nvidia.com/publication/scalable-gpu-graph-traversal)

This project is inspired by the aforementioned research paper and aims to demonstrate the advantages of utilizing CUDA for graph traversal tasks. Feel free to explore and adapt the code to your specific requirements and use cases.
