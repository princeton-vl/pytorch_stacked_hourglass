import os
import torch
import tqdm
import numpy as np
from data.VHS.vhs_loader import CoordinateDataset
from torch.utils.data import DataLoader
from PIL import Image

import cv2
import matplotlib.pyplot as plt


def do_inference(img_tensor, model):
    model.eval()
    # Move the input tensor to the same device as the model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    # Forward pass
    with torch.no_grad():
        preds = model(img_tensor)
    return preds

def draw_cross(img, center, color, size=1):
    """ Draw a small cross at the specified center point on the image. """
    x, y = center
    cv2.line(img, (x - size, y-size), (x + size, y+size), color, .5)
    cv2.line(img, (x+size, y - size), (x-size, y + size), color, .5)

def draw_predictions(img_tensor, pred_keypoints, true_points, config, save_path=None):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # CHW to HWC
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    nstack = pred_keypoints.shape[0]  # Number of stacks
    oup_dim = pred_keypoints.shape[1]  # Number of keypoints
    # preds are in range [0,63].  true_points are in range [0,1]
    scale_factor_pred = config['inference']['inp_dim'] / config['train']['output_res']
    scale_factor_true = config['inference']['inp_dim']
    # Draw predicted keypoints from each stack
    for stack in range(nstack):
        for k in range(oup_dim):
            x, y, conf = pred_keypoints[stack, k]
            #print(f"abc: {x}, {y}")
            x, y = int(x * scale_factor_pred), int(y * scale_factor_pred)
            draw_cross(img, (x, y), (0, 0, 255))  # Red color for predicted points

    # Draw true points
    true_points = true_points.squeeze(0)  # Remove batch dimension
    for point in true_points:
        x, y = point
        #print(f"efg: {x}, {y}")
        x, y = int(x * scale_factor_true), int(y * scale_factor_true)
        draw_cross(img, (x, y), (0, 255, 0))  # Green color for true points

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off the axis

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free memory

    return img

# #print(heatmaps.shape)  # torch.Size([bs, nstack, oup_dim, output_res, output_res])
#                           torch.Size([1, 2, 6, 64, 64])
def extract_keypoints_from_heatmaps(config, heatmaps):
    nstack = config['inference']['nstack']
    oup_dim = config['inference']['oup_dim']  # Number of keypoints
    output_res = config['train']['output_res']

    # Flatten the last two dimensions and find the index and max value of the highest point
    heatmaps = heatmaps.view(nstack, oup_dim, -1)
    maxval, idx = torch.max(heatmaps, dim=2)

    # Convert the 1D index to 2D coordinates
    x = (idx % output_res).view(nstack, oup_dim, 1)
    y = (idx // output_res).view(nstack, oup_dim, 1)
    maxval = maxval.view(nstack, oup_dim, 1)

    # Concatenate the x, y coordinates and max value
    keypoints = torch.cat((x, y, maxval), dim=2)
    return keypoints


def main():
    from train import init, reload
    
    opt = train.parse_command_line()  # Correctly reference 'train' for the function
    task, config = init(opt)

    pretrained_model_path = '/content/drive/MyDrive/point_localization/exps/hg2_real/checkpoint_2.133e-05_8.pt'
    if os.path.isfile(pretrained_model_path):  # Correctly check if the pretrained model exists
        print("=> loading pretrained model '{}'".format(pretrained_model_path))
        checkpoint = torch.load(pretrained_model_path)
        state_dict = {k.replace('model.module.', 'model.'): v for k, v in checkpoint['state_dict'].items()}
        config['inference']['net'].load_state_dict(state_dict, strict=False)  # Set strict=False if the model architectures are not exactly the same

    test_dir = '/content/drive/MyDrive/point_localization/VHS-Top-5286-Eric/Test'
    
    im_sz = config['inference']['inp_dim']
    heatmap_res = config['train']['output_res']
    test_dataset = CoordinateDataset(root_dir=test_dir, im_sz=im_sz, testing=True,\
                        output_res=heatmap_res, augment=False, only10=config['opt']['only10'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = config['inference']['net']
    model.eval()  # Set the model to evaluation mode

    for i, (img_tensor, true_points) in enumerate(test_loader):
        preds = do_inference(img_tensor, model)
        pred_keypoints = extract_keypoints_from_heatmaps(config, preds)

        scale_down_factor = 1.0/config['train']['output_res']
        pred_keypoints_scaled = pred_keypoints.clone()
        pred_keypoints_scaled[:, :, :2] *= scale_down_factor

        mse = torch.mean((pred_keypoints_scaled - true_points.clone()[0]) ** 2).item()

        save_dir = '/content/drive/MyDrive/point_localization/exps/'
        save_path = os.path.join(save_dir, f'img_{i}.png')
        
        draw_predictions(img_tensor[0], pred_keypoints, true_points, config, save_path=save_path)

        print(f"MSE for image {save_path}: {mse}")

if __name__ == '__main__':
    main()
