# 🧠 AI Projects

Welcome to my collection of ML and Deep Learning projects.

---

## 📚 Libraries Used

- **TensorFlow**
- **Keras**
- **Pandas**
- **Matplotlib**
- **NumPy**
- **OpenCV**
- **ImageDataGenerator**
- **Pillow**
- **Scikit Learn**
- **LangChain**
- **FAISS-CPU / GPU**
- **Surprise**
- **SpeechRecognition**
- **OpenAI**

**Kindly Find The `requirements.txt` in The Repo For Guided Installation via:**  
`pip install -r requirements.txt`

---

## ✅ Classic Machine Learning Projects

### 1. **Gender Classification from Names** 🧔👩  
Predicts a person's gender from their name.  
- **Task:** Binary Gender Classification (0 = Male, 1 = Female)  
- **Dataset:** [Gender Names Dataset](https://www.kaggle.com/datasets/gracehephzibahm/gender-by-name)  

| Metric               | Accuracy |
|-----------------------|----------|
| Training Accuracy     | 82.55%   |
| Validation Accuracy   | 80.60%   |
| Testing Accuracy      | 80.68%   |
---
### 2. **Stock Price Prediction (Apple Inc.)** 📈🍏  
A regression-based machine learning model to predict the stock **closing price** of Apple (AAPL) using historical data from **Yahoo Finance (2015–2024)**.

---

## 📚 Libraries Used
- **pandas**
- **numpy**
- **matplotlib**
- **scikit-learn**
- **yfinance**

---

- **Task:** Time Series Regression  
- **Target:** `Close` price  
- **Features Used:** `Open`, `High`, `Low`, `Volume`, `Date`  
- **Dataset Source:** [Yahoo Finance – AAPL](https://finance.yahoo.com/quote/AAPL/history?p=AAPL)  

- **ML Algorithms:**  
  - `LinearRegression`  

- **Preprocessing Steps:**  
  - Datetime parsing and sorting  
  - Feature selection  
  - Train/Test split using `train_test_split`  
  - Optional feature scaling (e.g., MinMaxScaler)

- **Evaluation Metrics:**  
  - **R² Score** – Measures how well the model explains variance in the target  
  - **Mean Squared Error (MSE)** – Average of squared prediction errors  
  - **Mean Absolute Error (MAE)** – Average of absolute prediction errors  

> 📌 **Note:**  
> R² Score is not the same as classification accuracy.  
> - R² ranges from −∞ to 1, with **1 meaning perfect prediction**  
> - Accuracy applies to classification, not regression tasks

- **Output:**  
  - Predictive model for Apple closing prices  
  - Visualization of Actual vs Predicted prices  
  - Average daily prediction error (e.g., $0.02 USD)

---
### 3. **Egyptian Tourism Recommender System Via Chatbot** 🏺🎮🇪🇬  
A comprehensive AI-powered tourism recommendation platform for Egypt featuring collaborative filtering, content-based recommendations, and an AR gamification layer to redistribute tourism flows and enhance cultural engagement.

- **Technologies Used:**  
  - [Surprise Library](https://surprise.readthedocs.io/) for collaborative filtering (SVD, KNN)  
  - [LangChain](https://python.langchain.com/) for LLM integration and RAG pipeline  
  - [OpenAI API (GPT-4o)](https://platform.openai.com/docs/guides/gpt) for natural language recommendations  
  - [FAISS](https://github.com/facebookresearch/faiss) for semantic similarity search & Vector Stores
  - [Scikit-learn](https://scikit-learn.org/) for content-based filtering and feature engineering  
  - **Pandas**, **NumPy**, **Matplotlib** for data processing and visualization

- **Dataset:**  
  - **1,000 synthetic users** with demographic profiles along with 10000 ratings, each user rates 10 movies from total of 20.
  - **32 links of Egyptian Ministry of Tourism and Antiquities sites** (Museums, Monuments, Archaeological Sites, Sunken Monuments, etc) using BeautifulSoap Scraping  
  - **10,000 user-place interactions** with ratings (2-5 scale) 

- **ML Techniques:**  
  - **Collaborative Filtering:** SVD (RMSE: 0.615, MAE: 0.41) and KNN (RMSE: 0.471, MAE:0.29)  
  - **Content-Based Filtering:** One-hot encoding + Cosine similarity

#### SVD
| Metric | Value |
|--------|-------|
| RMSE   | 0.615 |
| MAE    | 0.41  |

#### KNN
| Metric | Value |
|--------|-------|
| RMSE   | 0.471 |
| MAE    | 0.29  |


- **Key Features:**  
  - **Personalized Recommendations:** AI-driven destination suggestions based on user preferences  
  - **Tourism Redistribution:** Smart algorithms to promote lesser-known heritage sites  
  - **LLM Integration:** Natural language explanations for recommendations via RAG and detailed info providing

- **Problem Solved:**  
  - Addresses tourism overcrowding at major sites.
  - Promotes under-utilized heritage locations across Egypt  
  - Creates economic opportunities for secondary tourism sites
 
## ✅ Deep Learning Projects


### 1. **Dog Breed Classification** 🐶  
A Convolutional Neural Network (CNN) model trained to classify dog breeds from images.  
- **Technique:** CNN  
- **Dataset:** [Dog Breeds Dataset](https://www.kaggle.com/datasets/mohamedchahed/dog-breeds)  
- **Evaluation:** Accuracy, Confusion Matrix  

---

### 2. **Facial Emotion Recognition** 😊😢😠  
#### 1. AFFECT_NET Dataset( (RGB)
A model that detects human facial expressions and classifies them into emotional states.  
- **Emotions:** Happy, Sad, Angry, Surprise, Neutral, etc.  
- **Dataset:** [AffectNET](https://www.kaggle.com/datasets/mstjebashazida/affectnet)
- Had a challenge, the SOTA is 66%, we achieved (This Dataset was too large!):
  
| Metric                | Accuracy |
|-----------------------|----------|
| Training Accuracy     | 64.75%   |
| Validation Accuracy   | 63.70%   |

#### 2. FER2013+ Dataset (Grayscale)
A model that detects human facial expressions and classifies them into emotional states. 
- **Emotions:** Happy, Sad, Angry, Disgust, Neutral, Fear.  
- **Dataset:** [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
  
| Metric                | Accuracy |
|-----------------------|----------|
| Training Accuracy     | 86.75%   |
| Validation Accuracy   | 83.70%   |


---

### 3. **Gender Classification from Images** 📷🧔👩  
A CNN-based model to classify gender from facial images.  (0 => Females, 1 => Males)
- **Technique:** CNN for image-based classification  
- **Dataset:** [UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new)

| Metric                | Accuracy |
|-----------------------|----------|
| Training Accuracy     | 84.35%   |
| Validation Accuracy   | 85.60%   |
| Testing Accuracy      | 85%      |

---

### 4. **Dogs vs Cats Classification** 🐶🐱  
A binary image classification project using **VGG16** pretrained model with fine-tuning to distinguish between dog and cat images.  
- **Technique:** Transfer Learning with VGG16  
- **Dataset:** [Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)  
- **Approach:** Feature extraction + fine-tuning
  
| Metric                | Accuracy |
|-----------------------|----------|
| Training Accuracy     | 97.55%   |
| Validation Accuracy   | 95.60%   |
| Testing Accuracy      | 96.68%   |

---
### 5. **Real-Time Sign Language Detection** 🤟🎥  
An end-to-end system that recognizes sign language gestures in real time using computer vision and deep learning techniques. This project aims to bridge communication gaps for hearing-impaired individuals by providing robust, real-time gesture recognition from video input.

- **Key Tasks & Workflow:**
  - **Hand Detection (from scratch):**  
    - Leveraged the EgoHands dataset (subset of 1,000 annotated images)  
    - Implemented custom feature extraction methods:  
      - **Canny edge detection, Sobel gradients, contour & convex hull analysis**
      - **Local Binary Patterns (LBP)** for hand texture  
      - **Skin color masking** using HSV color space
      - **Histogram of Oriented Gradients (HOG)**
    - Trained a bespoke Convolutional Neural Network (CNN)  
    - Output: Binary hand detection and bounding box regression
  - **Sign Detection Using MediaPipe:**  
    - Used the **WL-ASL Dataset**: 2,000 words/letters, >21k videos  
    - Downloaded and organized missing video data; applied data augmentation (flipping, rotation, scaling)
    - Extracted frames at 5 FPS to create gesture image datasets
    - Detected hand(s) and located 21 key landmarks per hand using **MediaPipe**
    - Features: flattened landmark arrays for each frame
    - Built a Fully Connected Neural Network (FCNN) to classify gestures
    - Integrated sentence mapping using OpenCV and post-processing to ensure grammatical output
      
| Metric                | Accuracy |
|-----------------------|----------|
| Training Accuracy     | 86.55%   |
| Validation Accuracy   | 84.40%   |

  - **Alternative SIFT-Based Approach (Very Poor Performance):**  
    - Extracted SIFT keypoints/descriptors from detected hands  
    - Fed into an augmented FCNN for classification  
    - Noted for being less robust/noisier than MediaPipe pipeline

- **Datasets Used:**  
  - **WL-ASL** (Word-Level American Sign Language): >21,000 labeled videos, isolated sign focus
  - **EgoHands:** 48,000+ annotated hand images (subset used for training custom hand detector)

- **Technologies & Libraries:**  
  - **TensorFlow, Keras, OpenCV, Scikit-Learn, Pandas, NumPy**
  - **MediaPipe** for real-time hand and landmark detection
  - **PyTube** for dataset augmentation/downloading
  - **scikit-learn**
  - **Matplotlib** for plotting images
  - **NumPY**

 ### 6. **Brain Tumor Multi-Classification** 🧠   
Multi-class brain tumor classification from MRI scans using a custom deep convolutional neural network.

- **Task:** Classify MRIs into **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**.
- **Dataset:** [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Model:**  
  • 4 Convolutional blocks (with Dropout and MaxPooling)  
  • Dense layers (Dropout, L2-regularization)  
  • Trained with **AdamW** optimizer, data augmentation, early stopping, learning rate scheduling.
- **Techniques:** Heavy augmentation, callbacks for best performance, aggressive regularization for generalization.
- **Notable Steps:**  
  • Rigorous data visualization and balancing  
  • Confusion matrix for result analysis  
  • Prediction visualization on random MRI scans
  
| Metric                | Accuracy |
|-----------------------|----------|
| Training Accuracy     | 95.15%   |
| Validation Accuracy   | 90.60%   |
| Testing Accuracy      | 93.50%   |


 ### 7. **Distracted Driver Detection** 🧠   
A production-ready deep learning system for real-time driver distraction detection using multimodal fusion architecture combining **EfficientNet-B3** visual encoding with **YOLOv8-Pose** skeletal feature extraction, achieving **99.15% validation accuracy** on the State Farm dataset.

- **Task:** Classify drivers into 10 classes (c0-9).
- **Dataset:** Download from Kaggle](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data)
## Architecture

### Multimodal Fusion Design

The system employs a **dual-branch neural network** that processes both visual and skeletal information:

```
Input Driver Image (256x256)
    │
    ├─→ [EfficientNet-B3 CNN Branch]
    │      └─→ Feature Extraction (1536-dim)
    │
    └─→ [YOLOv8-Pose Detection Branch]
           └─→ 17 Keypoints Extraction
           └─→ MLP Processing (17×3 → 64-dim)
    │
    ├─→ [Fusion Layer]
    │      └─→ Concatenate (1536 + 64 = 1600-dim)
    │
    └─→ [Classification Head]
           └─→ Dense Layers (1600 → 512 → 256 → 10 classes)
```

  
**Key Performance Metrics:**
- Validation Accuracy: **99.15%**
- Architecture: Multimodal Fusion (EfficientNet-B3 + YOLOv8-Pose)
- Inference Speed: 10-20 FPS (GPU), 2-5 FPS (CPU)
- Model Size: ~45 MB


## ✅ LLM & GenAI Projects


### 1. **CodeGPT (Coding Assistant LLM)** 🤖🧑‍💻  
A fully functional, RAG-based coding assistant that leverages advanced LLMs for interactive programming help, retrieval-augmented generation, and real conversational memory—plus support for both text and speech!

- **Technologies Used:**  
  - [LangChain](https://python.langchain.com/) for workflow orchestration & conversational memory  
  - [ConversationalMemoryBuffers](https://python.langchain.com/docs/modules/memory/) to retain chat history and code context  
  - [OpenAI API (GPT-4o)](https://platform.openai.com/docs/guides/gpt) for code generation and dialog  
  - [FAISS](https://github.com/facebookresearch/faiss) for fast vector store and retrieval (semantic code/doc search)  
  - [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) for Speech-to-Text (STT)  
  - [OpenAI Whisper](https://github.com/openai/whisper)  

- **Key Features:**  
  - **Retrieval-Augmented Generation (RAG):**  
    Provides LLM answers empowered by code/document search from custom/local sources using FAISS.
  - **Conversational Memory:**  
    Remembers multi-turn programming conversations for context-aware help.
  - **Text & Voice IO:**  
    Ask coding questions via microphone, hear and read stepwise answers.
  - **Practical Use Cases:**  
    Code explanation, error fixing, inline suggestions, and knowledge base Q&A.
  - **Easy Extensibility:**  
    Plug in your own indexed docs, project files, or wiki for tailored assistance.


### 2. **Egyptian Tourism Recommender System Via Chatbot** 🏺🎮🇪🇬  
A comprehensive AI-powered tourism recommendation platform for Egypt featuring collaborative filtering, content-based recommendations, and an AR gamification layer to redistribute tourism flows and enhance cultural engagement.

- **Technologies Used:**  
  - [Surprise Library](https://surprise.readthedocs.io/) for collaborative filtering (SVD, KNN)  
  - [LangChain](https://python.langchain.com/) for LLM integration and RAG pipeline  
  - [OpenAI API (GPT-4o)](https://platform.openai.com/docs/guides/gpt) for natural language recommendations  
  - [FAISS](https://github.com/facebookresearch/faiss) for semantic similarity search & Vector Stores
  - [Scikit-learn](https://scikit-learn.org/) for content-based filtering and feature engineering  
  - **Pandas**, **NumPy**, **Matplotlib** for data processing and visualization

- **Dataset:**  
  - **1,000 synthetic users** with demographic profiles along with 10000 ratings, each user rates 10 movies from total of 20.
  - **32 links of Egyptian Ministry of Tourism and Antiquities sites** (Museums, Monuments, Archaeological Sites, Sunken Monuments, etc) using BeautifulSoap Scraping  
  - **10,000 user-place interactions** with ratings (2-5 scale) 

- **ML Techniques:**  
  - **Collaborative Filtering:** SVD (RMSE: 0.615) and KNN (RMSE: 0.471)  
  - **Content-Based Filtering:** One-hot encoding + Cosine similarity

- **Key Features:**  
  - **Personalized Recommendations:** AI-driven destination suggestions based on user preferences  
  - **Tourism Redistribution:** Smart algorithms to promote lesser-known heritage sites  
  - **LLM Integration:** Natural language explanations for recommendations via RAG and detailed info providing

- **Problem Solved:**  
  - Addresses tourism overcrowding at major sites.
  - Promotes under-utilized heritage locations across Egypt  
  - Creates economic opportunities for secondary tourism sites

### 3. **VEXA — macOS Virtual Assistant** 🖥️🎙️⚡  
A lightweight, always-on virtual assistant for macOS that detects a wake word and executes system-level commands. VEXA can open applications, terminate processes, run scripts, adjust system settings, and more — all via voice commands. The assistant is designed to be privacy-first and run entirely locally, with optional AI-enhanced capabilities.

- **Technologies Used:**  
  - [OpenAI API (GPT-4o)](https://platform.openai.com/docs/guides/gpt) for natural language understanding and responding with the json file scraped to be a command.
  - [Subprocess](https://docs.python.org/3/library/subprocess.html) for executing macOS commands  
  - [speech_recognition](https://pypi.org/project/SpeechRecognition/) for audio capture and offline transcription (Google) 
  - [rapidfuzz](https://github.com/maxbachmann/RapidFuzz) for fuzzy text matching to improve wake-word and command detection  

- **Key Features:**  
  - **Wake Word Activation:** Always listening for a customizable hotword (“Vexa” by default)  
  - **Command Execution:** Open apps, kill processes, adjust system volume, control music, or run custom scripts  
  - **Offline Operation:** Uses local speech recognition for privacy and low latency  
  - **Extensible Commands:** Easily add new functions with a Python function registry  
  - **Cross-Integration:** Supports AppleScript and macOS-native automation for advanced workflows

- **Problem Solved:**  
  - Eliminates the need for cloud-dependent assistants for local tasks  
  - Speeds up daily macOS interactions without mouse/keyboard  
  - Provides a developer-friendly, modifiable voice assistant framework

- **Example Use Cases:**  
  - “Vexa, open Safari” → Instantly launches Safari  
  - “Vexa, close Spotify” → Terminates the Spotify process  
  - “Vexa, volume up” → Increases system volume by 10%  

The following json is an example of many that the LLM generates:

```json
{
  "User": "Open Chrome.",
  "action": "open_app",
  "target": "Google Chrome",
  "value": null,
  "confirmation": false
}
```
This will be converted into a command  => open -a 'Google Chrome'









