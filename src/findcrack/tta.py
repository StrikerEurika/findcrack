import torch

def tta_forward(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    apply 4-way test-time augmentation and averages the sigmoid predictions.
    expects image tensor to be shape (C, H, W). returns (H, W)
    """
    
    with torch.no_grad():
        x = image_tensor.unsqueeze(0)

        # original
        original_prediction = torch.sigmoid(model(x))
        
        # horizontal flip
        horizontal_flip_prediction = torch.flip(torch.sigmoid(model(torch.flip(x, dims=[3]))), dims=[3])

        # vertical flip
        vertical_flip_prediction = torch.flip(torch.sigmoid(model(torch.flip(x, dims=[2]))), dims=[2])

        # diagonal flip
        # diagonal_flip_prediction = torch.flip(torch.sigmoid(model(torch.flip(x, dims=[2, 3]))), dims=[2, 3])

        # average the predictions
        # final_prediction = (original_prediction + horizontal_flip_prediction + vertical_flip_prediction + diagonal_flip_prediction) / 4

        # rotate 90 degrees clockwise
        rotated_90_prediction = torch.rot90(
            torch.sigmoid(model(torch.rot90(x, k=1, dims=[2, 3]))),
            k=-1, dims=[2,3]
        )
        
        # average predictions
        averaged_prediction = (
            original_prediction + 
            horizontal_flip_prediction + 
            vertical_flip_prediction + 
            rotated_90_prediction) / 4.0

    return averaged_prediction.squeeze(0)