# Runtime Client

To request inferences.

```bash
cp runtime_client/test/test_request.py ./

python test_request_multiprocessing.py \
  --client_type='kfserving' \
  --domain='http://localhost:28080' \
  --model_id='model' \
  --concurrency=100 \
  --batch_size=1 \
  --data_shape='(224, 224, 3)' \
  --each_data_num=20
```