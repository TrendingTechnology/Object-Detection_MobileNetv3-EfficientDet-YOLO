# Object Detection: YOLO, MobileNetv3 and EfficientDet

Object detection using OpenCv and Tensroflow with a serverless API on Google Cloud Run.




# Run Notebooks
clone the repo and run 

```
docker run --rm -it -p 8888:8888 -v $(pwd):/app  imadelh/opencv_tf:base jupyter lab --ip 0.0.0.0 --no-browser --allow-root
```

Jupyter Lab will be accessible at 0.0.0.0:8888 and you can run notebooks for inference (in ./artifacts/) for each model.

# Run API

To run the API for object detection, you have to use the docker image that contains the pre-trained weights (imadelh/opencv_tf:full).

docker run --rm -it -p 8080:8080 imadelh/opencv_tf:full


The API can used as follows 

http://0.0.0.0:8080/predict/v1?model=NAME-OF-MODEL&image_url=IMAGE-URL

Where NAME-OF-MODEL& can be : yolo, mobilenet or efficientdet
IMAGE-URL : a direct URL to an image

For example

http://0.0.0.0:8080/predict/v1?model=yolo&image_url=https://imadelhanafi.com/data/draft/random/img2.jpg

Will result in 

[{"bbox":[137.0,187.0,96,144],"classe":"cat","confidence":0.9843610525131226}]


#
