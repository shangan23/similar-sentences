[![PyPI version](https://badge.fury.io/py/similar-sentences.svg)](https://badge.fury.io/py/similar-sentences)

# similar-sentences

Install the package

```python
pip install similar-sentences
```

```python
from SimilarSentences import SimilarSentences
model = SimilarSentences('model.zip',"predict")
text = 'How are you doing?'
simple = model.predict(text, 2, "simple")
detailed = model.predict(text, 2, "detailed")
print(simple)
print(detailed)
```

Output looks like,

```python
#simple output
["Did I get that right?", "Hi, how can I help?"]
#detailed output
[[{"sentence": "Did I get that right?", "score": 0.7910412874624612}], [{"sentence": "Hi, how can I help?", "score": 0.7824735035480156}]]
````
