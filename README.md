# Modelz Imagebind Template

## Build Docker Image

You can choose to use the one of the following method:

* `Dockerfile`
  * `docker buildx build -t <docker_hub_user_name>/<image_name> --push .`
* `build.envd`
  * `envd build -f :serving --output type=image,name=docker.io/<docker_hub_user_name>/<image_name> --push`

## How to use this template

This template based on `mosec`, see [client.py](www.github.com/tensorchord/modelz-imagebind/blob/main/example/client.py) to connect remote endpoint.

## Acknowledge

[facebookresearch/ImageBind](https://github.com/facebookresearch/ImageBind)
