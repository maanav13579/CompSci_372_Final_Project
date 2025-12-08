# Attribution

This document provides detailed attribution for all external resources, AI-generated code, libraries, and datasets used in this project.

## AI-Generated Sections

Portions of this project's code and documentation were generated with assistance from **Claude (Anthropic)**. The AI assistant was used for:

- Code
  - Web Demo using gradio
    - I DID NOT WRITE THE CODE FOR THIS DEMO. It is purely for having a nicer interface for the video, and I do not claim any points for a deployed web application (don't think it counts anyway).
  - Debugging and troubleshooting PyTorch compatibility issues
  - Training curve generation and visualization code
- Documentation
  - README.md
    - file structure
    - Evalutation and individual contributions sections
  - SETUP.md
    - File Structure
    - Debugging setup scripts 
  - ATTRIBUTION.md
    - File Structure
    - Citations for datasets and pre-trained models(see below)

All AI-generated code was reviewed, tested, and modified as needed to ensure correctness and fit project requirements.

## Datasets

### Food-101

- **Source:** ETH Zurich, Computer Vision Lab
- **URL:** https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
- **Citation:**
  ```
  Bossard, L., Guillaumin, M., & Van Gool, L. (2014). 
  Food-101 – Mining Discriminative Components with Random Forests. 
  European Conference on Computer Vision (ECCV).
  ```

### Nutrition5k

- **Source:** Google Research
- **URL:** https://github.com/google-research-datasets/Nutrition5k
- **Citation:**
  ```
  Thames, Q., Karber, A., Kuehl, O., Childers, J., & Walters, L. (2021).
  Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food.
  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
  ```
- Also used gsutil datadownload scripts from developers

## Pre-trained Models

### ResNet50

- **Source:** PyTorch / torchvision
- **Pre-trained on:** ImageNet (ILSVRC2012)
- **URL:** https://pytorch.org/vision/stable/models.html
- **Citation:**
  ```
  He, K., Zhang, X., Ren, S., & Sun, J. (2016).
  Deep Residual Learning for Image Recognition.
  Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
  ```



