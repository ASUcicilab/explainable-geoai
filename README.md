# explainable-geoai

## Overview 
This is the dataset and code for the work "Explainable GeoAI: Performance assessment of deep learning model visualization techniques to understand AI learning processes with geospatial data"

## Installation and Prerequisities
This project is built upon [PyTorch](https://pytorch.org) framework and its library [Captum](https://captum.ai) for model interpretability. To install both PyTorch and Captum, please check their official installation guide for more details. 

## Getting Start
### Training a New Model
This project used VGG-16 as the demo model. A trained model is saved under [`models`](./models). To train a new model, you can run:
```bash
python train_net.py 
```

### Generating Visualizations
To generate visualizations of a given image, you should specify the image name `<image_name>`. You can run:
```bash
python gen_figure.py --image-name <image_name>
# example:
python gen_figure.py --image-name crater_000001
```

After the generating process, you will have four images in the [`outputs`](./outputs) folder:
* `<image_name>.original.png`: the original image with bounding box labels
* `<image_name>.occ.png`: Occlusion visualization
* `<image_name>.cam.png`: Grad-CAM visualization
* `<image_name>.ig.png`: Integrated Gradients visualization

The file names of the figures in the manuscript are:

Figure 2: volcano_000039

Figure 3: meander_000059

Figure 4: crater_000083

Figure 5: dunes_000010

Figure 6: volcano_000055

Figure 7: icebergtongue_000067




