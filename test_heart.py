import os
import torch
import tqdm
import numpy as np
from data.VHS.vhs_loader import CoordinateDataset
from torch.utils.data import DataLoader
from PIL import Image

import cv2
import matplotlib.pyplot as plt

def extract_keypoints_from_heatmaps(config, heatmaps, output_res):
    nstack = config['inference']['nstack']
    oup_dim = config['inference']['oup_dim']
    heatmaps = heatmaps.view(nstack, oup_dim, -1)
    maxval, idx = torch.max(heatmaps, dim=2)
    maxval = maxval.view(nstack, oup_dim, 1)
    idx = idx.view(nstack, oup_dim, 1)
    keypoints = torch.cat((idx % output_res, idx // output_res, maxval), dim=2)
    return keypoints


def do_inference(img_tensor, model):
    model.eval()
    # Move the input tensor to the same device as the model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    # Forward pass
    with torch.no_grad():
        preds = model(img_tensor)
    return preds

def draw_predictions(img_tensor, keypoints, save_path=None):
    # Convert the tensor to an image
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # CHW to HWC
    img = (img * 255).astype(np.uint8)

    nstack = keypoints.shape[0]  # Number of stacks
    oup_dim = keypoints.shape[1]  # Number of keypoints
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Colors for each keypoint
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Adjust as needed
    print("drawing predictions")
    for stack in range(nstack):
        for k in range(oup_dim):
            x, y, conf = keypoints[stack, k].detach().cpu().numpy()
            x*= 256/64
            y*=256/64
            x, y = int(x), int(y)  # Convert to integer
            print(f"x: {x}, y: {y}")
            if conf > 0.2:  # Threshold for confidence, adjust as needed
                color = colors[k % len(colors)]  # Color for this keypoint
                cv2.circle(img, (x, y), 5, color, -1)  # Draw circle
    save_path = '/content/drive/MyDrive/point_localization/exps/\
    img4.png'
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Turn off the axis
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory
    # if save_path:
    #     cv2.imwrite(save_path, img)
    # else:
    #     # Display the image
    #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #     plt.show()

    return img

import matplotlib.pyplot as plt




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
    test_dir = '/content/drive/MyDrive/point_localization/VHS-Top-5286-Eric/Test'
    im_sz = config['inference']['inp_dim']
    heatmap_res = config['train']['output_res']
    test_dataset = CoordinateDataset(root_dir=test_dir, im_sz=im_sz, output_res=heatmap_res, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    def display_img_tensor(img_tensor):
        
        img_tensor = img_tensor.squeeze(0).squeeze(0)
        # Convert from CHW to HWC format if necessary
        if img_tensor.shape[0] == 3:
            img_tensor = img_tensor.permute(1, 2, 0)
        # Scale the pixel values back to 0-255 and convert to numpy array
        img_np = (img_tensor.numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        plt.imshow(img_pil, cmap='gray')
        plt.axis('off')  # Turn off the axis
        save_path = '/content/drive/MyDrive/point_localization/exps/\
        img2.png'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free memory




    model = config['inference']['net']
    model.eval()  # Set the model to evaluation mode

    for i, (img_tensor, _) in enumerate(test_loader):#enumerate(tqdm.tqdm(test_loader)):
        # display_img_tensor(img_tensor)
        #print(f"test image >>>>>>>>>>:\n{test_img}")
        #test_img = np.linspace(0, 255, 100*100).reshape((100, 100)).astype(np.uint8)
        # test_img = img_tensor.squeeze(0).cpu().numpy().transpose(1,2,0).astype(np.uint8)

        # plt.imshow(test_img, cmap='gray')
        # plt.axis('off')  # Turn off the axis
        #plt.show()
        # save_path = '/content/drive/MyDrive/point_localization/exps/plot2.png'
        # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        # plt.close()  # Close the figure to free memory
        
        if i >= 1:
            break

        # preds will be of shape [batch_size, nstack, oup_dim, height, width]
        preds = do_inference(img_tensor, model)
        #print(preds.shape)  # torch.Size([1, 2, 6, 64, 64])

        # Extract keypoints from the heatmaps
        # Assuming output resolution (height, width) of heatmaps is 64
        keypoints = extract_keypoints_from_heatmaps(config, preds, output_res=64)  # [0] to remove batch dimension
        # keypoints shape rn: torch.Size([2, 6, 3])

        # # Draw predictions on the image
        draw_predictions(img_tensor[0], keypoints)

if __name__ == '__main__':
    main()
