# similar-sentences

```python
model = SimilarSentences('model_org.zip')
text = 'How are you doing?'
suggestions = model.predict(text, 2, "simple")
print(suggestions)
```
