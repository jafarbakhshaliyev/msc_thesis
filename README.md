# Master Thesis: Leveraging Patch-Level Shuffling to Boost Generalization and Robustness in Time Series Analysis

This repository contains the implementation of novel data augmentation methods for time series analysis, specifically **Temporal Patch Shuffle (TPS)** and **Temporal Index Patch Shuffle (TIPS)**, with comprehensive experiments for both forecasting and classification tasks.

## Abstract

Data augmentation is a crucial technique for enhancing model generalization and robustness, particularly in deep learning models where training data is limited. While numerous augmentation methods exist for Time Series Classification, most are not directly applicable to Time Series Forecasting due to the need to maintain temporal coherence. 

This work proposes **Temporal Patch Shuffle (TPS)**, a general-purpose augmentation method applicable to both Time Series Forecasting and Classification. TPS extracts overlapping temporal patches from each time series and shuffles the less informative patches—identified via variance-based importance scoring—to introduce meaningful variability while preserving signal integrity. 

For classification tasks, we introduce **Temporal Index Patch Shuffle (TIPS)**, a variant of TPS that exchanges patches across samples of the same class instead of shuffling patches within a sample. Both methods are lightweight, easy to integrate, and compatible with any model architecture.

## Folder Structure

```
├── time_series_forecasting/    # Forecasting implementations and experiments
│   ├── utils/
        └── augmentations.py    # TPS and other augmentation methods
│   └── README.md               # Detailed forecasting setup and results
├── time_series_classification/ # Classification implementations and experiments
│   ├── MultiRocket\
         └── augmentations.py   # TPS, TIPS, and other augmentation methods  
│   └── README.md               # Detailed classification setup and results
├── latex/                  
│   └── master_thesis.pdf       # Complete thesis document
└── README.md                  
```

## Key Contributions

### Temporal Patch Shuffle (TPS)
- **Universal applicability**: Works for both forecasting and classification
- **Preserves temporal coherence**: Maintains signal integrity while introducing variability
- **Variance-based patch selection**: Targets less informative patches for shuffling
- **Lightweight implementation**: Easy integration with any model architecture

### Temporal Index Patch Shuffle (TIPS)
- **Classification-specific**: Exchanges patches across samples of the same class
- **Enhanced diversity**: Creates more varied training samples than TPS
- **Class-aware augmentation**: Maintains class boundaries 

## Experimental Results

### Forecasting Performance
- **7 long-term forecasting datasets** evaluated
- **4 short-term forecasting datasets** tested  
- **5 recent models**: TSMixer, DLinear, PatchTST, TiDE, and LightTS
- **Consistent improvements** across all model-dataset combinations

### Classification Performance
- **UCR and UEA repositories**: Univariate and multivariate datasets
- **MiniRocket and MultiRocket** models tested
- **TIPS outperforms** existing augmentation baselines and TPS
- **Comprehensive ablation studies** demonstrate effectiveness and robustness

## Getting Started

### For Implementation Details
Each folder contains comprehensive documentation and implementations:

- **[Forecasting Augmentations](./time_series_forecasting/utils/augmentations.py)** - TPS implementation and forecasting-specific augmentations
- **[Classification Augmentations](./time_series_classification/minirocket/src/augmentation.py)** - TPS, TIPS, and classification-specific augmentations

### Individual Folder Documentation
- **[time_series_forecasting/](./time_series_forecasting/)** - Complete setup, models, and forecasting experiments
- **[time_series_classification/](./time_series_classification/)** - Full implementation guide and classification experiments

### For Complete Documentation and Results
For detailed methodology, comprehensive experimental results, and analysis:

**📄 [Master Thesis PDF](./latex/master_thesis.pdf)**

The thesis document provides:
- Comprehensive literature review and related work
- Detailed TPS and TIPS methodology explanations
- Complete experimental setup
- Extensive results analysis and ablation studies
- Conclusions and future research directions

## Quick Navigation

| Component | Methods | Documentation |
|-----------|---------|---------------|
| Forecasting | TPS + other augmentations | [Folder](./time_series_forecasting/) \| [Augmentations](./time_series_forecasting/utils/augmentations.py) |
| Classification | TPS + TIPS + other augmentations | [Folder](./time_series_classification/) \| [Augmentations](./time_series_classification/minirocket/src/augmentation.py) |
| Complete Research | Full thesis documentation | [PDF](./latex/master_thesis.pdf) |

## Usage Workflow

1. **Start with the thesis PDF** to understand TPS/TIPS methodology and experimental design
2. **Explore augmentation implementations** in the respective augmentation folders
3. **Navigate to specific task folders** for complete experimental reproduction
4. **Follow individual README files** for detailed setup and execution instructions

## Method Highlights

- **📊 Extensive Evaluation**: 11 forecasting datasets, UCR/UEA classification benchmarks
- **🔬 Rigorous Testing**: Multiple model architectures and comprehensive ablations  
- **⚡ Practical Implementation**: Lightweight, model-agnostic, easy integration
- **📈 Consistent Gains**: Demonstrated improvements across diverse tasks and models
- **🎯 Methodological Innovation**: Novel temporal patch-based approach with variance scoring


## Acknowledgements

This work builds upon several research contributions and open-source implementations. For detailed acknowledgements, citations, and licensing information specific to each component, please refer to:

- **[Forecasting Acknowledgements](./time_series_forecasting/README.md#-acknowledgements)** - Dominant Shuffle, FrAug, and other forecasting methods
- **[Classification Acknowledgements for MiniRocket](./time_series_classification/minirocket/README.md#-acknowledgements)** and **[Classification Acknowledgements for MultiRocket](./time_series_classification/MultiRocket/README.md#-acknowledgements)**  - MiniRocket, MultiRocket, UCR/UEA datasets, and augmentation methods

**Note**: This README serves as a navigation guide. For detailed implementation instructions, methodological details, complete experimental results, and full acknowledgements, please refer to the individual folder READMEs, augmentation implementations, and the master thesis PDF document.
