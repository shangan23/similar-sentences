# similar-sentences

```python
model = SimilarSentences('model_org.zip')
text = 'Would you like to add TravelPass? The cost of TravelPass is $5/day for North America and $10/day for the rest of the world.'
suggestions = model.predict(text, 2, "simple")
print(suggestions)
```
