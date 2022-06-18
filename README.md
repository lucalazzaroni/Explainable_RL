# Docker example
This Dockerfile allows you to create a Docker image including the following packages:
- [highway-env](https://github.com/eleurent/highway-env) v.1.4
- [rl-agents](https://github.com/eleurent/rl-agents) v.1.0.dev0
- PyTorch 1.11.0 (supporting CUDA version 11.3)
- [SHAP](http://github.com/slundberg/shap)

After installation, try it with the provided Python file as follows:
```
docker run --gpus all image-id python main.py
```
Or interactively:
```
docker run -it --gpus all image-id /bin/bash

```
Alternatively, you can also directly download the pre-built image from Docker Hub :
```
docker pull lucalazzaroni/lucalazzaroni/highway_env_rl_agent:v00
```