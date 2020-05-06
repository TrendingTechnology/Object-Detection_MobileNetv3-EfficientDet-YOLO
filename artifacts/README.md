## Run on local machine
Clone the repo and run

```
docker run --rm -it -p 8888:8888 -v $(pwd):/app  imadelh/opencv_tf:base jupyter lab --ip 0.0.0.0 --no-browser --allow-root
```

Jupyter Lab will be accessible at http://127.0.0.1:8888 and you can run notebooks for inference (in ./artifacts/) for each model.

**NB**: make sure to download models artifacts for each model (uncomment cell where the model weights are downloaded).

## Models

- Yolo v3: inference using OpenCv
- MobileNetv3: inference using OpenCv
- EfficientDet: inference using Tensorflow

