# Device-Directed Speech Segmentation
<!--
<div align="center">
<!--
[![Paper](https://img.shields.io/badge/📄_Paper-arXiv-b31b1b.svg)](PAPER_LINK)
[![Dataset](https://img.shields.io/badge/🗂️_Dataset-Kaggle-20beff.svg)](https://www.kaggle.com/datasets/dung8204/device-directed-television-dataset-dtd)
[![Code](https://img.shields.io/badge/💻_Code-GitHub-171515.svg)](https://github.com/dungdt-infopstats/Device-Directed-Speech-Segmentation)
[![License](https://img.shields.io/badge/📋_License-MIT-green.svg)](https://github.com/dungdt-infopstats/Device-Directed-Speech-Segmentation/blob/main/LICENSE)

*A novel frame-level approach for isolating device-directed speech segments in virtual assistant applications*

</div>
-->
---

## Overview

Device-directed Speech Segmentation (DDSS) presents a promising approach to virtual assistant speech processing by moving beyond traditional utterance-level classification to precise frame-level segmentation. This method shows potential to significantly improve downstream ASR performance by isolating only the device-directed portions of speech.

## Key Features

- **Novel DDSS Framework** - Frame-level segmentation for precise device-directed speech isolation
- **Comprehensive Data Pipeline** - Reproducible synthesis pipeline for multiple VA domains
- **Public DTD Dataset** - First open dataset with 330+ hours of TV-domain commands
- **Multiple Model Variants** - Acoustic, ASR, and Fusion-based architectures

## Setup

> **📖 Documentation Reference**  
> Complete setup instructions are maintained in the project's documentation directory.  
> **Link**: [Setup Documentation](https://github.com/dungdt-infopstats/Device-Directed-Speech-Segmentation/tree/main/docs)
## Dataset

### Access
<div align="center">

[![Kaggle Dataset](https://img.shields.io/badge/📊_DTD_Dataset-Kaggle-20beff?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/dung8204/device-directed-television-dataset-dtd)

</div>

### Dataset Information
| **Property** | **Details** |
|--------------|-------------|
| **Size** | 330+ hours of TV-domain commands |
| **Split** | 80/20 train/test, no speaker overlap |
| **Scenarios** | 5 types: non-command, single-command, chain-command, single-mix, chain-mix |
| **Format** | WAV files + JSON annotations |
| **Domain** | Television commands (channels, movies, apps) |

### Sample Audio Examples

<details>
<summary>Click to see scenario examples</summary>

**Single-command:**
> "Turn on YouTube"

**Chain-command:** 
> "Open Netflix, search for action movies, play the first result"

**Single-mix:**
> "Turn on YouTube" (to device) + "What do you want to watch?" (to another person)

**Chain-mix:**
> "Open the weather app" + conversational speech + "Set volume to 50%"

</details>

## Results

| **Method** | **WER ↓** | **D ↓** | **I ↓** | **S ↓** | **D+S ↓** |
|------------|-----------|---------|---------|---------|-----------|
| No-filtering | 271.6% | 2.5% | 252.5% | 16.6% | 19.1% |
| Detection-only | 203.9% | 2.5% | 184.8% | 16.6% | 19.1% |
| **Fusion DDSS** | **47.8%** | **5.7%** | **29.5%** | **12.6%** | **18.3%** |
| Oracle DDSS | 22.4% | 3.5% | 8.4% | 10.5% | 14.0% |

## Citation


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<!--
## 👥 Contact

- **Tri Dung Do** - dungdt.research@gmail.com
---
<div align="center">
<i>Research conducted during internship at Voice Team - Viettel AI ❤️</i>
</div>
-->
