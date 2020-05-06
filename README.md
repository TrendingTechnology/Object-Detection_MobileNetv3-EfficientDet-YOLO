# Object Detection: YOLO, MobileNetv3 and EfficientDet

Object detection using OpenCv and Tensroflow with a serverless API on Google Cloud Run.

- Blog post: https://imadelhanafi.com/posts/object_detection_yolo_efficientdet_mobilenet/

- Live version: `https://vision.imadelhanafi.com/predict/v1?model=MODEL_NAME&image_url=URL` where MODEL_NAME is `yolo` or `mobilenet`.

    example:
    ```
    https://vision.imadelhanafi.com/predict/v1?model=mobilenet&image_url=https://imadelhanafi.com/data/draft/random/img4.jpg
    #Returns:
    [{"bbox":[114,17,186,222],"confidence":0.853282630443573,"label":"bear"}]
  
    https://vision.imadelhanafi.com/predict/v1?model=yolo&image_url=https://imadelhanafi.com/data/draft/random/img2.jpg
    #Returns:
    [{"bbox":[137.0,187.0,96,144],"confidence":0.9843610525131226,"label":"cat"}]
    ```

## Run Notebooks
Clone the repo and run 

```
docker run --rm -it -p 8888:8888 -v $(pwd):/app  imadelh/opencv_tf:base jupyter lab --ip 0.0.0.0 --no-browser --allow-root
```

Jupyter Lab will be accessible at http://127.0.0.1:8888 and you can run notebooks for inference (in ./artifacts/) for each model.

## Run API

To run the API for object detection, you have to use the docker image that contains the pre-trained weights (imadelh/opencv_tf:full).

```
docker run --rm -it -p 8080:8080 imadelh/opencv_tf:full
```

The API can used as follows 
```
http://0.0.0.0:8080/predict/v1?model=NAME-OF-MODEL&image_url=IMAGE-URL
```

Where NAME-OF-MODEL can be: `yolo`, `mobilenet` or `efficientdet` and IMAGE-URL is a direct URL to an image

Example:
```
http://0.0.0.0:8080/predict/v1?model=yolo&image_url=https://imadelhanafi.com/data/draft/random/img2.jpg

Returns:
[{"bbox":[137.0,187.0,96,144],"confidence":0.9843610525131226,"label":"cat"}]
```

This docker image (imadelh/opencv_tf:full) can be deployed to a cloud instance or serverless container services like Google Cloud Run. 
Steps are explained in details here: https://github.com/imadelh/NLP-news-classification#serverless-deployement---google-run

Serverless version may suffer from cold-start if the service does not receive requests for a long time.

---
Imad El
