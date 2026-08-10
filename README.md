# NLP Sarcasm Detector: A Hands-On Intro to Machine Learning

This project is a simple introduction to Machine Learning (ML) and Natural Language Processing (NLP). Instead of using common beginner examples like house-price prediction, this project teaches a computer to identify whether a sentence is sarcastic or sincere.

The model is trained on a small set of labelled examples and learns patterns that help distinguish sarcastic statements from genuine ones. To do this, the text is converted into numbers using TF-IDF Vectorization, and a Logistic Regression model is then used to make predictions. The model can take a new sentence and predict whether it is sarcastic, along with a confidence score.

## Project Structure

```text
nlp-sarcasm-detector/
├── .gitignore
├── requirements.txt
├── main.py
└── README.md
```

## Creating the Repository from Scratch

First, create a new GitHub repository called `nlp-sarcasm-detector` and clone it to your computer.

```bash
git clone <repository-url>
cd nlp-sarcasm-detector
```

Create and activate a virtual environment.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Create the following files:

```text
.gitignore
requirements.txt
main.py
README.md
```

Add the following contents to `.gitignore`:

```text
venv/
__pycache__/
*.pyc
*.png
```

Add the following *ontents to `requirements.txt`:

``*text
numpy
scikit-learn
matplotlib*```

Paste the provided Python cod* into `main.py`.

Install the requ*red packages:

```bash
pip install*-r requirements.txt
```

Run the p*oject:

```bash
python main.py
```*
When finished, commit and push yo*r work to GitHub:

```bash
git add*.
git commit -m "Add NLP sarcasm d*tector project"
git push
```

## Expected Output

After running the s*ript, the model will:

- Train on *arcastic and sincere examples.
- D*splay important words and phrases *earned during training*
* Predict whether new sentences are*sarcastic or sincere.
- Generate a*chart called `feature_weights.png`*showing the most influential featu*es.

Example prediction:

```text
*nput: "Oh fantastic, another delay*on the subway today"
Prediction: S*RCASTIC

Input: "I am so grateful *or all your hard work on this proj*ct"
Prediction: SINCERE
```

## Technologies Used

- Python
- NumPy
-*Scikit-learn
- Matplotlib
- Natura* Language Processing (NLP)

## Learning Outcome

This project demonst*ates the basic machine-learning*workflow of preparing data, traini*g a model, making predictions, and*interpreting results. It provides * simple and practical introduction*to NLP and text classification.
``*