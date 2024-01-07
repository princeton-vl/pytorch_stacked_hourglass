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

def draw_predictions(img_tensor, keypoints, config, save_path=None):
    # Convert the tensor to an image
    img = img_tensor.permute(1,2,0).cpu().numpy()  # CHW to HWC
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)

    nstack = keypoints.shape[0]  # Number of stacks
    oup_dim = keypoints.shape[1]  # Number of keypoints
    scale_factor = config['inference']['inp_dim'] / config['train']['output_res']
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Colors for each keypoint
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Adjust as needed
    print("drawing predictions")
    for i, stack in enumerate(range(nstack)):

        for k in range(oup_dim):
            x, y, conf = keypoints[stack, k]
            x, y = int(x), int(y)  # Convert to integer
            print(f"x: {x}, y: {y}")
            x = int(x * scale_factor)
            y = int(y * scale_factor)

            color_intensity = int(conf * 255)
            red = int(conf * 255) if i==0 else 0
            blue = int(conf * 255) if i==1 else 0

            # Draw the circle with color intensity based on confidence
            cv2.circle(img, (x, y), 3, (red, 0, blue), -1)  # Green with varying intensity

    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Turn off the axis
    
    if save_path:
        print(f"getting here")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free memory
    # if save_path:
    #     cv2.imwrite(save_path, img)
    # else:
    #     # Display the image
    #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #     plt.show()
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
    from train import init
    func, config = init()

    # # ERIC: Assuming the path to your pretrained model
    # pretrained_model_path = '/content/drive/MyDrive/point_localization/stacked_hourglass_point_localization/exp/hourglass_01/checkpoint.pt'
    # config['opt'].pretrained_model = pretrained_model_path

    opt = config['opt']
    if opt.pretrained_model and os.path.isfile(opt.pretrained_model):  # Check if pretrained model path is provided
        print("=> loading pretrained model '{}'".format(opt.pretrained_model))
        checkpoint = torch.load(opt.pretrained_model)
        # state_dict = {k.replace('model.module.', 'model.'): v for k, v in checkpoint['state_dict'].items()}
        # config['inference']['net'].load_state_dict(state_dict)
        state_dict = {k.replace('model.module.', 'model.'): v for k, v in checkpoint['state_dict'].items()}
        config['inference']['net'].load_state_dict(state_dict, strict=True)

    # Load or initialize the model's parameters
    # hg_dir = '/Users/ewern/Desktop/code/MetronMind/stacked_hourglass_point_localization/'
    # chkpt_path = os.path.join(hg_dir, 'models/local_models/checkpoint.pt')
    test_dir = '/content/drive/MyDrive/point_localization/VHS-Top-5286-Eric/Test'
    
    im_sz = config['inference']['inp_dim']
    heatmap_res = config['train']['output_res']
    test_dataset = CoordinateDataset(root_dir=test_dir, im_sz=im_sz, output_res=heatmap_res, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = config['inference']['net']
    model.eval()  # Set the model to evaluation mode

    for i, (img_tensor, _) in enumerate(test_loader):#enumerate(tqdm.tqdm(test_loader)):
        if i >= 1:
            break

        # preds will be of shape [batch_size, nstack, oup_dim, height, width]
        preds = do_inference(img_tensor, model)
        #print(preds.shape)  # torch.Size([1, 2, 6, 64, 64])

        # Extract keypoints from the heatmaps
        # Assuming output resolution (height, width) of heatmaps is 64
        keypoints = extract_keypoints_from_heatmaps(config, preds)  # [0] to remove batch dimension
        # keypoints shape rn: torch.Size([2, 6, 3])

        # # Draw predictions on the image
        #save_path = '/content/drive/MyDrive/point_localization/exps/img8.png'
        save_path = '/content/drive/MyDrive/point_localization/exps/img8.png'
        draw_predictions(img_tensor[0], keypoints, config, save_path=save_path)
        print(f"ballsssss: {img_tensor.shape}")

if __name__ == '__main__':
    main()
