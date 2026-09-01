\# TruthLens AI 🔎 



\### BERT-Based Fake News Detection System



TruthLens AI is an NLP-based fake news classification system built using \*\*BERT (Bidirectional Encoder Representations from Transformers)\*\* and \*\*PyTorch\*\*.



The project takes a news article or claim as input and predicts whether it is more likely to be \*\*REAL\*\* or \*\*FAKE\*\* based on patterns learned from a labeled news dataset.



> ⚠️ TruthLens AI is a machine-learning classification project and should not be considered a replacement for professional fact-checking or trusted news sources.



\---



\## 🚀 Features -



\- 🧠 BERT-based text classification

\- 📰 Fake vs. real news prediction

\- 🔤 BERT tokenization and preprocessing

\- ⚡ PyTorch model training

\- 📊 Training and validation evaluation

\- 🎯 Probability-based predictions

\- 💻 Command-line prediction interface

\- 🧪 Dataset and model testing utilities



\---



\## 🧠 How It Works -



The overall pipeline is:



```text

News Article / Claim

&#x20;       ↓

Text Preprocessing

&#x20;       ↓

BERT Tokenizer

&#x20;       ↓

Token IDs + Attention Mask

&#x20;       ↓

BERT Model

&#x20;       ↓

Classification Layer

&#x20;       ↓

Fake / Real Probabilities

&#x20;       ↓

Final Prediction





TruthLens AI uses a pretrained bert-base-uncased model and adapts it for binary text classification.



📊 Model Performance -



During model evaluation, TruthLens AI achieved:



Validation Accuracy: 96.61%



This result represents performance on the validation dataset used during training.



Model performance on completely new, unseen real-world claims may differ. Accuracy should therefore be interpreted in the context of the dataset and evaluation methodology.



🛠️ Technologies Used -



Python

PyTorch

Hugging Face Transformers

BERT

Pandas

NumPy

scikit-learn

tqdm



📁 Project Structure -

TruthLens-AI/

│

├── app/                    # Application-related files

│

├── data/                   # Dataset files

│

├── docs/                   # Project documentation

│

├── models/                 # Trained model files

│

├── notebooks/              # Exploratory notebooks

│

├── screenshots/            # Project screenshots

│

├── src/

│   ├── dataloader.py       # Data loading utilities

│   ├── dataset.py          # Dataset preparation

│   ├── evaluate.py         # Model evaluation

│   ├── explore\_dataset.py  # Dataset exploration

│   ├── model.py            # Model architecture

│   ├── predict.py          # Interactive prediction

│   ├── preprocess.py       # Text preprocessing

│   ├── test\_dataset.py     # Dataset tests

│   ├── test\_model.py       # Model tests

│   ├── tokenizer\_test.py   # Tokenizer testing

│   ├── train.py            # Model training

│   └── train\_model.py      # Training utilities

│

├── tests/                  # Additional tests

│

├── .gitignore

├── LICENSE

├── README.md

└── requirements.txt



⚙️ Installation - 



Clone the repository:



git clone git@github.com:VIMAL360/TruthLens-AI.git

cd TruthLens-AI



Create a virtual environment:



python -m venv .venv



Activate it on Windows PowerShell:



.\\.venv\\Scripts\\Activate.ps1



Install dependencies:



pip install -r requirements.txt

▶️ Running the Detector



Run:



python src/predict.py



Enter a news article or claim when prompted.



Example:



============================================================

&#x20;         TruthLens AI - Fake News Detector

============================================================



Enter a news article:



NASA announced that Earth will experience six hours of

complete darkness tomorrow because of a rare cosmic alignment.



Example output:



============================================================



Real probability : 24.91%

Fake probability : 75.09%

Prediction : FAKE NEWS



============================================================

🏋️ Training-



The model can be trained using the training pipeline provided in:



src/train.py



The training process includes:



Dataset loading

Tokenization

BERT model initialization

Mini-batch training

Loss calculation

Validation

Accuracy evaluation

Model weight saving



tqdm is used to display training progress and batch information.



🔬 Evaluation-



The project includes utilities for evaluating the trained model and testing the dataset/model components.



Relevant files include:



src/evaluate.py

src/test\_dataset.py

src/test\_model.py



💡 Why BERT? -



Traditional machine-learning approaches often rely heavily on manually engineered text features.



BERT uses transformer-based attention mechanisms to understand relationships between words within their surrounding context.



For example:



"The company denied the report."



"The report denied the company."



The meaning changes depending on the relationship between the words.



BERT is designed to capture these contextual relationships, making it useful for NLP classification tasks.



📌 Limitations-



TruthLens AI is a text classification model, not a live fact-checking engine.



It does not automatically:



Verify claims against the internet

Check government databases

Determine whether a source is trustworthy

Detect every form of misinformation

Guarantee factual correctness



A high validation accuracy does not mean that every prediction made on unseen real-world information will be correct.



For important information, predictions should be verified using reliable sources and professional fact-checking organizations.



🔮 Future Improvements -



Possible future improvements include:



🌐 Real-time web-based fact verification

📰 News-source credibility analysis

🔗 URL and source verification

📚 Larger and more diverse datasets

🎯 Better handling of short claims and headlines

📈 Precision, recall and F1-score analysis

🧪 Larger unseen test datasets

🌍 Multilingual fake-news detection

🖥️ Web-based user interface

⚡ Model optimization for faster inference

👨‍💻 Project Goal



The goal of TruthLens AI was to build a practical NLP project that demonstrates how transformer-based models such as BERT can be applied to binary text classification.



The project covers the complete machine-learning workflow:



Data

&#x20;↓

Preprocessing

&#x20;↓

Tokenization

&#x20;↓

Model Training

&#x20;↓

Validation

&#x20;↓

Prediction

&#x20;↓

Evaluation





⭐ If you found this project interesting, feel free to explore the code and experiment with the model.

