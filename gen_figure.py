import argparse
import copy
import os
import xml.etree.ElementTree as et

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients, LayerGradCam, Occlusion
from PIL import Image
from torchvision import models


def gen_figure(img_name):
    # dataset class
    class_names = ['crater', 'dunes', 'hill', 'icebergtongue', 'lake', 'meander', 'river', 'volcano']
    cls_to_idx = {'crater': 0, 'dunes': 1, 'hill': 2, 'icebergtongue': 3, 'lake': 4, 'meander': 5, 'river': 6, 'volcano': 7}
    

    # transformation 
    transform = transforms.Compose([transforms.Resize([224, 224]),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    

    # load image
    img_dir = './datasets/natural_feature_dataset/'
    img_path = os.path.join(img_dir, 'images', img_name + '.jpg')
    img = Image.open(img_path)
    width, height = img.size
    img = img.convert('RGB')
    img = transform(img)
    img = img[None, :]


    # load trained model
    model = models.vgg16(pretrained=True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_ftrs, 8)
    model.load_state_dict(torch.load('./models/vgg16_best.pth'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    img = img.to(device)
    model.eval()


    # run prediction on the image 
    gt_label =  img_name.split('_')[0]    
    gt_target = cls_to_idx[gt_label]
    
    m = torch.nn.Softmax(dim=1)
        
    with torch.no_grad():
        output = m(model(img))
        gt_prob = output[0, gt_target].item()
        predict_target = torch.argmax(output)
        predict_prob = output[0, predict_target].item()
        predict_label = class_names[predict_target]
        print('running prediction on {}...'.format(img_name))
        print('Label: {} ({})'.format(gt_label, gt_prob))
        print('Prediction: {} ({})'.format(predict_label, predict_prob))


    # draw original images
    fig, ax = plt.subplots()
    inp = copy.deepcopy(img)
    inp = np.transpose(inp.squeeze().cpu().detach().numpy(), (1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    ax.imshow(inp)
    ax.axis('off')
        
    # draw labels
    tree = et.parse(os.path.join(img_dir, 'labels', img_name + '.xml'))
    root = tree.getroot()   
    wratio = 224.0 / width
    hratio = 224.0 / height
        
    for bb in root.iter('bndbox'):
        xmin = int(bb[0].text) * wratio 
        ymin = int(bb[1].text) * hratio
        xmax = int(bb[2].text) * wratio 
        ymax = int(bb[3].text) * hratio        
        rec = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='b', fill=False)
        ax.add_patch(rec)

    plt.savefig('./outputs/{}_original.png'.format(img_name), bbox_inches='tight', dpi=300)
    plt.close()


    # Grad-CAM
    gridspec = {'width_ratios': [1,1,0.1]}
    fig, ax = plt.subplots(1,3,figsize=(11,4.7),gridspec_kw=gridspec)
    layer_gc = LayerGradCam(model, model.features[30])
    attri_gc = layer_gc.attribute(img, predict_target)
    cam = attri_gc.squeeze().cpu().detach().numpy()
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  
    cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
    cam = np.uint8(Image.fromarray(cam).resize((224, 224), Image.ANTIALIAS))/255
    heatmap = ax[0].imshow(cam, cmap='jet')
    ax[0].axis('off')
    
    # Grad-CAM on the original image
    ax[1].imshow(inp)
    ax[1].axis('off')
    ax[1].imshow(cam, cmap='jet', alpha=0.3)
    plt.colorbar(heatmap, cax=ax[2])
    plt.savefig('./outputs/{}_cam.png'.format(img_name), bbox_inches='tight', dpi=300)
    plt.close()


    # occlusion
    gridspec = {'width_ratios': [1,1,0.1]}
    fig, ax = plt.subplots(1,3,figsize=(11,4.7),gridspec_kw=gridspec)
    occ = Occlusion(model)
    attri_occ = occ.attribute(img, target=predict_target, sliding_window_shapes=(3,40,40), show_progress=True)
    attri_occ = np.transpose(attri_occ.squeeze().cpu().detach().numpy(), (1,2,0))
    attri_occ = attri_occ[:,:,0]
    attri_occ = (attri_occ - np.min(attri_occ)) / (np.max(attri_occ) - np.min(attri_occ))
    heatmap = ax[0].imshow(attri_occ, cmap='jet')
    ax[0].axis('off')
    
    # occlusion on the image
    ax[1].imshow(inp)
    ax[1].axis('off')
    ax[1].imshow(attri_occ, cmap='jet', alpha=0.3)
    plt.colorbar(heatmap, cax=ax[2])
    plt.savefig('./outputs/{}_occ.png'.format(img_name), bbox_inches='tight', dpi=300)
    plt.close()


    # Integrated Gradients
    gridspec = {'width_ratios': [1,1,0.1]}
    fig, ax = plt.subplots(1,3,figsize=(11,4.7),gridspec_kw=gridspec)
    ig = IntegratedGradients(model)
    attri_ig = ig.attribute(img, target=predict_target)
    attri_ig = np.transpose(attri_ig.squeeze().cpu().detach().numpy(), (1,2,0))
    attri_ig = np.sum(attri_ig, axis=2)
    attri_ig = np.abs(attri_ig)
    attri_sorted = np.sort(attri_ig.flatten())
    attri_ig = (attri_ig < attri_sorted[-50]) * attri_ig
    attri_ig = (attri_ig - np.min(attri_ig)) / (np.max(attri_ig) - np.min(attri_ig))
    heatmap = ax[0].imshow(attri_ig, cmap='jet')
    ax[0].axis('off')
    
    # ig on the image
    ax[1].imshow(inp)
    ax[1].axis('off')
    ax[1].imshow(attri_ig, cmap='jet', alpha=0.3)
    plt.colorbar(heatmap, cax=ax[2])
    plt.savefig('./outputs/{}_ig.png'.format(img_name), bbox_inches='tight', dpi=300)
    plt.close()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-name",
        type=str,
        help="image file name",
        required=True,
    )
    
    return parser

if __name__=="__main__":
    args = get_parser().parse_args()
    gen_figure(args.image_name)







