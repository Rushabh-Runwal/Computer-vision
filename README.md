# 🖼️ Computer Vision Assignment: Supervised Contrastive Learning, Transfer Learning, and Zero-Shot Models

This repository contains Google Colab notebooks and detailed video walkthroughs demonstrating key computer vision techniques across **supervised contrastive learning**, **transfer learning (across modalities)**, and **zero-shot learning**.

The project is divided into **four parts**, each covering specific learning paradigms with clear examples, visualizations, and outputs.

---
## 🎥 [Video Walkthrough](youtube.com)
---

## 📦 Repository Overview

- [**Part 1:** Supervised Contrastive Learning]
- [**Part 2:** Transfer Learning on Images, Video, Audio, and NLP]
- [**Part 3:** Zero-Shot Transfer Learning (CLIP, BigTransfer)]
- [**Part 4:** Vision Classifiers on MNIST, FashionMNIST, CIFAR-10, X-ray, and CT Scans]

---

## 📚 References & Hints

- [Contrastive Loss for Supervised Classification (article)](https://towardsdatascience.com/contrastive-loss-for-supervised-classification-224ae35692e7)  
- [Keras Supervised Contrastive Learning Example](https://keras.io/examples/vision/supervised-contrastive-learning)  
- [TensorFlow Hub Transfer Learning Tutorials](https://amitness.com/2020/02/tensorflow-hub-for-transfer-learning)  
- [CLIP Zero-Shot Classifier](https://towardsdatascience.com/how-to-try-clip-openais-zero-shot-image-classifier-439d75a34d6b)  
- [Keras SOTA Models](https://keras.io/examples/vision/bit)  
- [Hands-on ML Notebooks](https://github.com/ageron/handson-ml3/blob/main/14_deep_computer_vision_with_cnns.ipynb)

---

## 🚀 Project Parts and Notebooks

---

### 🟡 **Part 1: Supervised Contrastive Learning**

- **Notebook:** `part1_supervised_contrastive_vs_softmax.ipynb`
- Tasks:
  - Implement supervised contrastive loss-based classification.
  - Compare performance against softmax-based classification.
  - Provide clear visualizations of embeddings and decision boundaries.

References:
- [Colab examples](https://docs.google.com/presentation/d/1UxtHDwjViC7VpSb0zB-kajGQ-TwznQmc-7LsbHRfO3s/edit#slide=id.gcdc5f16e5b_20_5)
- [Keras contrastive learning](https://keras.io/examples/vision/supervised-contrastive-learning)

---

### 🟡 **Part 2: Transfer Learning Across Modalities**

- **Notebooks:**  
  - `part2_audio_transfer_learning_yamnet.ipynb`  
  - `part2_video_transfer_learning_movienet.ipynb`  
  - `part2_nlp_transfer_learning_tfhub.ipynb`  
  - `part2_image_transfer_learning_catsdogs.ipynb`

- Tasks:
  - Apply transfer learning using pretrained models:
    - As **feature extractors**.
    - With **fine-tuning**.
  - Cover use cases across:
    - Audio (YAMNet).
    - Video (Action Recognition).
    - NLP (Text Classification).
    - Images (Cats vs. Dogs, Dog Breeds).

References:
- [Audio Transfer Learning](https://blog.tensorflow.org/2021/03/transfer-learning-for-audio-data-with-yamnet.html)
- [Video Action Recognition](https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub)
- [NLP Text Classification](https://www.tensorflow.org/hub/tutorials/tf2_text_classification)
- [Image Transfer Learning](https://www.tensorflow.org/tutorials/images/transfer_learning)

---

### 🟡 **Part 3: Zero-Shot Transfer Learning**

- **Notebooks:**  
  - `part3_clip_zero_shot.ipynb`  
  - `part3_bigtransfer_tfhub.ipynb`

- Tasks:
  - Demonstrate zero-shot classification using **CLIP**.
  - Apply transfer learning using **BigTransfer (BiT)** or other SOTA models from TF Hub.

References:
- [CLIP Zero-Shot Guide](https://towardsdatascience.com/how-to-try-clip-openais-zero-shot-image-classifier-439d75a34d6b)
- [BigTransfer on TFHub](https://keras.io/examples/vision/bit)

---

### 🟡 **Part 4: Vision Classifiers & SOTA Models**

- **Notebooks:**  
  - `part4_mnist_fashionmnist_cifar10.ipynb`  
  - `part4_xray_pneumonia_classification.ipynb`  
  - `part4_3d_ct_scan_classification.ipynb`

- Tasks:
  - Train classifiers on:
    - MNIST, FashionMNIST, CIFAR-10 datasets.
    - Chest X-ray (pneumonia detection).
    - 3D CT scans.
  - Apply transfer learning using:
    - EfficientNet.
    - BiT (BigTransfer).
    - SOTA models (MLP-Mixer, ConvNeXt v2).

References:
- [X-ray Classification with TPUs](https://keras.io/examples/vision/xray_classification_with_tpus)
- [3D Image Classification](https://keras.io/examples/vision/3D_image_classification)
- [EfficientNet Fine-Tuning](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning)
- [MLP Image Classification](https://keras.io/examples/vision/mlp_image_classification)

---
