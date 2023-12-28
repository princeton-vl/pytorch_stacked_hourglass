import os
import torch
import tqdm
import numpy as np
from data.VHS.vhs_loader import CoordinateDataset
from torch.utils.data import DataLoader

import cv2
import matplotlib.pyplot as plt

def do_inference(img_tensor, model):
    model.eval()
    # Move the input tensor to the same device as the model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    # Add a batch dimension to img_tensor since models expect batched input
    img_tensor = img_tensor.unsqueeze(0)
    # Forward pass
    with torch.no_grad():  # No need to track gradients during inference
        preds = model(img_tensor)
    # Remove batch dimension and move predictions to CPU for further processing
    preds = preds.squeeze(0).cpu()
    return preds

def draw_predictions(img_tensor, preds, save_path=None):
    # Convert the tensor to an image
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # CHW to HWC
    img = (img * 255).astype(np.uint8)

    # Draw each prediction
    for point in preds:
        x, y = point
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

    if save_path:
        cv2.imwrite(save_path, img)
    else:
        # Display the image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    return img

def main():
    from train import init
    func, _, _, config = init()

    # Assuming the path to your pretrained model
    pretrained_model_path = '/content/drive/MyDrive/point_localization/stacked_hourglass_point_localization/exp/hourglass_01/checkpoint.pt'
    config['opt'].pretrained_model = pretrained_model_path

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



    model = config['inference']['net']
    model.eval()  # Set the model to evaluation mode

    for i, (img_tensor, _) in enumerate(tqdm.tqdm(test_loader)):
        if i >= 20: break

        preds = do_inference(img_tensor, func)
        # Assuming preds are in the format: Nx2 (for N keypoints)
        # Convert tensor to numpy array and reshape
        preds = preds.view(-1, 2).numpy()

        # Draw predictions on the image and print/show them
        draw_predictions(img_tensor[0], preds)

if __name__ == '__main__':
    main()
