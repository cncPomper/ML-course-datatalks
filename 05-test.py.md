import requests

```python
url = 'http://localhost:9696/predict'
```


```python
customer = {
  "job": "management", 
  "duration": 400, 
  "poutcome": "success"
}
```


```python
requests.post(url, json=customer).json()
```




    {'client': True, 'y_pred': 0.7590966516879658}




```python
client = {"job": "student", "duration": 280, "poutcome": "failure"}
requests.post(url, json=client).json()
```




    {'client': False, 'y_pred': 0.33480703475511053}




```python

```
