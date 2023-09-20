## Introduction

This repository is a resource collection for Keyword Spotting (KWS), also known as Hotword or Wakeword Detection. KWS is used extensively in voice-activated personal assistants such as Alexa and Google Assistant. 
The Repo includes a list of research papers, models, implementations, algorithms, and tutorials related to KWS.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Research Papers](#research-papers)
3. [Models](#models)
4. [Implementations](#implementations)
5. [Algorithms](#algorithms)
6. [Tutorials](#tutorials)
7. [Tools and Libraries](#tools-and-libraries)
9. [ToDo](#todo)

## Getting Started

To get familiar with KWS, there needs to be a basic understanding of digital signal processing and machine learning. Here are some resources

- 📖 [Introduction to Digital Signal Processing](https://www.dspguide.com/pdfbook.htm)
- 📖 [Machine Learning Yearning by Andrew Ng](https://github.com/ajaymache/machine-learning-yearning/)

## Research Papers

- EfficientWord-Net
	- 📑 [Paper](https://arxiv.org/pdf/2111.00379.pdf)
	- 📦 [Repository](https://github.com/Ant-Brain/EfficientWord-Net)
- MatchboxNet
	- 📑 [Paper](https://arxiv.org/pdf/2004.08531.pdf)
	- 📦 [Repository](https://github.com/Rumeysakeskin/Speech-Command-Recognition)
- Streaming KWS on Mobile Devices
	- 📑 [Paper](https://arxiv.org/pdf/2005.06720.pdf)
	- 📦 [Repository](https://github.com/google-research/google-research/tree/master/kws_streaming)

## Models
-  WakeNet by ESP
	- 📘 [Documentation](https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/wake_word_engine/README.html)
-  Deep Residual Networks for KWS
	- 📘 [Documentation](https://arxiv.org/pdf/1711.07128.pdf)
	- 📦 [Repository](https://github.com/ARM-software/ML-KWS-for-MCU)

## Implementations

- Keyword Transformer
	- 📦 [Repository](https://github.com/ARM-software/keyword-transformer)
- KWS Streaming
	- 📦 [Repository](https://github.com/google-research/google-research/tree/master/kws_streaming)
	- 📘 [Inference on ESP32](https://medium.com/@dmytro.korablyov/first-steps-with-esp32-and-tensorflow-lite-for-microcontrollers-c2d8e238accf)

## Algorithms

- Beamforming
	- 📑 [Paper](https://invensense.tdk.com/wp-content/uploads/2015/02/Microphone-Array-Beamforming.pdf)
- Dynamic Time Warping
	- 📑 [Paper](https://www.aaai.org/Papers/Workshops/1994/WS-94-03/WS94-03-031.pdf)

## Tutorials

-  [Building Your Own KWS Model](https://edgeimpulse.com/blog/tutorial-keyword-spotting)
-  [TensorFlow Lite](https://www.tensorflow.org/lite/examples/)

## Tools and Libraries

-  TensorFlow Lite for Microcontrollers
	- 📘 [Documentation](https://www.tensorflow.org/lite/microcontrollers)
-  PocketSphinx
	- 📘 [Documentation](https://cmusphinx.github.io/wiki/tutorialpocketsphinx/)

## ToDo

- Add accuracy and parameter diagrams for Broadcasted Residual Learning paper.
- Add performance table for PhonMatchNet paper.
