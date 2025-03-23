# Addressing Class Imbalance in Credit Fraud Detection using Synthetic Data Generation


## Overview

This project tackles the common challenge of class imbalance in credit card fraud detection by leveraging advanced synthetic data generation techniques. Using TabDDPM (Tabular Denoising Diffusion Probabilistic Models) and TVAE (Triple Variational Autoencoder), we generate synthetic samples for the minority class (fraudulent transactions) to create a balanced dataset for improved machine learning performance.

## Problem Statement

Credit card fraud detection datasets typically suffer from extreme class imbalance - in our case, only 0.172% (492 out of 284,807) transactions are fraudulent. This imbalance leads to biased models that favor the majority class and perform poorly on detecting actual fraud cases.

## Solution Approach

Our methodology involves:

1. **Data Preparation**: We use the publicly available credit card fraud dataset from Kaggle, containing transactions from European cardholders.

2. **Synthetic Data Generation**: We employ two neural network-based models:
   - **TabDDPM**: A diffusion model specialized for tabular data
   - **TVAE**: A variational autoencoder approach for synthetic data generation

3. **Cross-Validation Strategy**: We implement 10-fold cross-validation, ensuring synthetic data is only generated for the minority class in the training folds.

4. **Classification**: We train Random Forest classifiers on both the original imbalanced data and the synthetically balanced data.

5. **Performance Evaluation**: We assess model performance using multiple metrics optimized for imbalanced classification.

## Evaluation Metrics

We evaluate our approach using metrics specifically chosen for imbalanced classification problems:

- **F1-Score**: Harmonic mean of precision and recall
- **Kappa Score**: Agreement between predicted and actual classes
- **Average Precision Score**: Area under the precision-recall curve
- **G-Mean**: Geometric mean of sensitivity and specificity
- **Accuracy**: Overall correctness of predictions

## Project Structure

```
├── final/
│   ├── iisertvm_final_tapddpm.ipynb
│   └── iisertvm_final_tvae.ipynb       
├── report
└── README.md               # Project overview
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- scikit-learn 1.0+
- pandas, numpy, matplotlib

### Installation

```bash
git clone https://github.com/yourusername/credit-fraud-synthetic-data.git
cd credit-fraud-synthetic-data
pip install -r requirements.txt
```

### Dataset

The credit card fraud dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/mlg_ulb/creditcardfraud).


### Running the Experiments

1. Run the respective Jupyter Notebook to get the results.

## Results

Our experiments show that balancing the dataset with synthetic samples significantly improves the model's ability to detect fraudulent transactions. Detailed results and visualizations can be found in the Jupyter Notebook.

## Future Work

- Explore additional synthetic data generation techniques
- Implement feature engineering methods to enhance model performance
- Investigate ensemble techniques for improved fraud detection
- Apply the methodology to other domains with class imbalance challenges

## References

- [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg_ulb/creditcardfraud)
- [TabDDPM Paper](https://arxiv.org/abs/2209.15421)
- [TVAE Paper](https://arxiv.org/abs/1802.04403)
- [SDV Project Documentation](https://docs.sdv.dev/sdv/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dr. Saptrishi Bej, School of Data Science, IISER Thiruvananthapuram, Kerela, India
- The original authors of the credit card fraud dataset
- The researchers behind TabDDPM and TVAE models
- The Synthetic Data Vault (SDV) project community
