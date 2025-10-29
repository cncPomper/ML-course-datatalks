# Conf

```bash
conda create -n ml python=3.12
conda activate ml

# install package
pip3 install --upgrade pip
pip3 install -e .
```

## Building an image
```bash
cp Dockerfile_base Dockerfile
docker build -t mlzoomcamp -f Dockerfile .
```

```bash
docker run -it --rm -p 8000:8000 mlzoomcamp
```