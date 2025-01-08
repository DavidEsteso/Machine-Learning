# Convolutional Neural Network Implementation and Analysis

## Project Overview
This project implements and analyzes various aspects of Convolutional Neural Networks (CNNs) using Python, TensorFlow, and Keras. It explores different CNN architectures, investigates the effects of various hyperparameters, and demonstrates practical applications in image processing and classification.

## Key Features
- Custom 2D convolution implementation from scratch
- Image processing with different convolution kernels
- CNN architecture design and optimization
- Analysis of model performance with varying:
  - Training data sizes (5k to 40k samples)
  - L1 regularization weights
  - Network architectures (shallow vs deep)
  - Pooling strategies

## Technical Implementation

### Part 1: Image Processing
- Custom implementation of 2D convolution operation
- Support for different padding types ('same' and 'valid')
- Configurable stride parameters
- Image processing with custom kernels for feature extraction

### Part 2: CNN Architecture
- Modular CNN implementation with configurable parameters
- Support for different model architectures:
  - Basic CNN with stride-based downsampling
  - MaxPooling variant
  - Deeper, thinner network variant
- Comprehensive model evaluation and visualization

## Results and Analysis
- Detailed performance comparisons across different model configurations
- Analysis of training vs. validation metrics
- Investigation of overfitting mitigation strategies
- Execution time analysis for different model architectures

## Installation and Usage

### Prerequisites
- Python 3.x
- TensorFlow 2.x
- NumPy
- Pillow
- Matplotlib
- scikit-learn

### Running the Project
```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py
```

## Project Structure
```
project/
├── main.py           # Main execution script
├── part1.py          # Convolution implementation
├── part2.py          # CNN implementation and training
├── data/            # Dataset directory
└── output/          # Results and visualizations
```

## Demonstrated Skills

### Technical Skills
1. **Deep Learning & Computer Vision**
   - CNN architecture design
   - Image processing fundamentals
   - Feature extraction techniques

2. **Programming & Software Development**
   - Python programming
   - Object-oriented design
   - Clean code practices
   - Modular architecture

3. **Data Science & Analysis**
   - Performance metrics analysis
   - Data visualization
   - Statistical analysis
   - Hyperparameter optimization

4. **Machine Learning**
   - Model training and evaluation
   - Regularization techniques
   - Cross-validation
   - Performance optimization

### Analytical Skills
1. **Problem Solving**
   - Systematic approach to optimization
   - Performance bottleneck identification
   - Trade-off analysis

2. **Research & Analysis**
   - Comprehensive performance evaluation
   - Detailed comparative analysis
   - Results interpretation
   - Scientific methodology

3. **Technical Writing**
   - Clear documentation
   - Complex concept explanation
   - Results visualization and presentation

### Project Management
1. **Organization**
   - Structured project layout
   - Clear code organization
   - Systematic experimentation

2. **Documentation**
   - Comprehensive README
   - Code comments
   - Technical documentation

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- David Esteso Calatrava for the course structure and guidance
- School of Computer Science and Statistics for the learning opportunity