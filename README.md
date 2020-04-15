[![PyPI version](https://badge.fury.io/py/similar-sentences.svg)](https://badge.fury.io/py/similar-sentences)

# Similar Sentence Prediction with acurate results

## Getting Started

Install the package

```python
pip install similar-sentences
```

## Train the model with your dataset

Prepare your dataset and save the content to `sentences.txt`

```
Hi, thanks for contacting.
Hello there!
Hi there, welcome!
Hi, how can I help?
In a few words, how can help?
Hi again, welcome back.
Hi! Welcome back.
Good morning! 
Good afternoon! 
Good evening! 
Good morning! Welcome.
Good afternoon! Welcome.
Good evening! Welcome.
Hello, how can I help?
Welcome.
Welcome back.
Thanks for contacting.
Goodbye!
Thanks for contacting. Goodbye!
Thanks for contacting. Bye!
Happy to help!
Glad I could help!
```

Supply the sentences to build the model.

```python
from SimilarSentences import SimilarSentences
# Make sure the extension is .txt
model = SimilarSentences('sentences.txt',"train")
model.train()
```
The code snipet will produce model.zip.

## Predicting from your model

Load the model.zip from the training.

```python
from SimilarSentences import SimilarSentences
model = SimilarSentences('model.zip',"predict")
text = 'Hi.How are you doing?'
simple = model.predict(text, 2, "simple")
detailed = model.predict(text, 2, "detailed")
print(simple)
print(detailed)
```

Output looks like,

```python
#simple output
[
  "Hello there! Did I get that right?",
  "Right Hi, how can I help?"
]

#detailed output
[
  [
    {
      "sentence": "Hello there!",
      "score": 0.938870553799856
    },
    {
      "sentence": "Did I get that right?",
      "score": 0.7910412586610753
    }
  ],
  [
    {
      "sentence": "Right",
      "score": 0.9161810654762793
    },
    {
      "sentence": "Hi, how can I help?",
      "score": 0.7824734658953297
    }
  ]
]
````
<sub>The package depends on `sentence-transformers` python package</sub>

:+1: :sparkles: :camel: :tada: :rocket: :metal: :octocat:  HAPPY CODING :octocat: :metal: :rocket: :tada: :camel: :sparkles: :+1:
