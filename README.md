[![PyPI version](https://badge.fury.io/py/similar-sentences.svg)](https://badge.fury.io/py/similar-sentences) [![Python 3](https://img.shields.io/badge/python-3.0-blue.svg)](https://www.python.org/downloads/release/python-300/)

# Similar sentence Prediction with more accurate results with your dataset on top of BERT pertained model.

## Setup

Install the package

```python
pip install similar-sentences
```

### Methods to know

#### SimilarSentences(FilePath,Type)
  * **FilePath**: Reference to model.zip for prediction. Reference to sentences.txt for training.  
  * **Type**: `predict` or `train`

#### .train(PreTrainedModel)
 * Used for training the setences. Which required `(".txt", "train")` as parameter in SimilarSentences
 * **PreTrainedModel (optional)**: Any of the below model can be passed for training,by default #1 will be applied
 1. bert-base-nli-mean-tokens: BERT-base model with mean-tokens pooling. Performance: STSbenchmark: 77.12
 2. bert-base-nli-max-tokens: BERT-base with max-tokens pooling. Performance: STSbenchmark: 77.21
 3. bert-base-nli-cls-token: BERT-base with cls token pooling. Performance: STSbenchmark: 76.30
 4. bert-large-nli-mean-tokens: BERT-large with mean-tokens pooling. Performance: STSbenchmark: 79.19
 5. bert-large-nli-max-tokens: BERT-large with max-tokens pooling. Performance: STSbenchmark: 78.41
 6. bert-large-nli-cls-token:  BERT-large with CLS token pooling. Performance: STSbenchmark: 78.29
 7. roberta-base-nli-mean-tokens: RoBERTa-base with mean-tokens pooling. Performance: STSbenchmark: 77.49
 8. roberta-large-nli-mean-tokens: RoBERTa-base with mean-tokens pooling. Performance: STSbenchmark: 78.69
 9. distilbert-base-nli-mean-tokens: DistilBERT-base with mean-tokens pooling. Performance: STSbenchmark: 76.97  
 [More details](https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/nli-models.md#pre-trained-models)

#### .predict(InputSentences, NumberOfPrediction, DesiredJsonOutput)
  * Used for predicting the setences. Which required `(".zip", "predict")` as parameter in SimilarSentences  
  * **InputSentences**: To find the similar sentence for.   
  * **NumberOfPrediction**: Number of results for the prediction  
  * **DesiredJsonOutput**: The output will be in JSON format. `simple` produces a plain output. `detailed` produces detailed output with score 
  
#### .reload()
  * Used for reloading (or) updating the model. Which required `(".zip", "predict")` as parameter in SimilarSentences

#### .batch_predict(BatchFile,NumberOfPrediction)
  * Used for reloading (or) updating the model. Which required `(".zip", "predict")` as parameter in SimilarSentences
  * **BatchFile**: Batch file with sentences to predict, has to be in .txt format.   
  * **NumberOfPrediction**: Number of results for the prediction  
  
## Getting Started

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
:+1: :sparkles: :camel: :tada: :rocket: :metal: :octocat:  HAPPY CODING :octocat: :metal: :rocket: :tada: :camel: :sparkles: :+1:
