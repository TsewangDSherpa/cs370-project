from transformers import SamModel, SamConfig, SamProcessor
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def Predict(filename):
    # Load the model configuration
    model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # Create an instance of the model architecture with the loaded configuration
    sidewalkModel = SamModel(config=model_config)
    # Update the model by loading the weights from the saved file.
    sidewalkModel.load_state_dict(torch.load("./best_model.pth", map_location=torch.device('cpu')))  # Load model on CPU

    # set the device to cpu since on hugging face using FREE CPU
    device = "cpu"
    sidewalkModel.to(device)

    # Define the custom colormap
    cmap_data = [(0, 0, 0, 0), (1, 1, 1, 0), (255/255, 255/255, 10/255, 1)]
    custom_cmap = LinearSegmentedColormap.from_list("BlackToTransparent", cmap_data)

    # Function to generate a bounding box covering the full image
    def get_full_image_bounding_box(image):
        height, width = image.shape[:2]
        return [0, 0, width, height]  # Format: [x_min, y_min, x_max, y_max]

        
    # Load image and ground truth segmentation
    input_image_path = "./UploadedImg/" + filename
    input_image = Image.open(input_image_path)

    # Resize the image to 256x256
    resize_transform = transforms.Resize((256, 256))
    resized_image = resize_transform(input_image)
    # Convert the resized image to RGB format
    resized_image_rgb = resized_image.convert("RGB")

    # Convert the RGB image to a numpy array
    resized_image_np = np.array(resized_image_rgb)

    # Generate bounding box covering the full image
    full_image_box = get_full_image_bounding_box(np.array(resized_image_np))

    # Prepare image + box prompt for the model
    inputs = processor(resized_image_rgb, input_boxes=[[full_image_box]], return_tensors="pt")

    # Move the input tensor to the CPU
    inputs = {k: v.cpu() for k, v in inputs.items()}

    sidewalkModel.eval()

    # Forward pass
    with torch.no_grad():
        outputs = sidewalkModel(**inputs, multimask_output=False)

    # Apply sigmoid
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    # Convert soft mask to hard mask
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg = (medsam_seg_prob > 0.65).astype(np.uint8)

    # Save the resulting image
    result_image_path = "./Result/" + filename
    plt.imshow(np.array(resized_image_rgb))  # Assuming the image is RGB
    plt.imshow(medsam_seg, cmap=custom_cmap, alpha=1)  # Overlay predicted segmentation
    plt.axis('off')  # Turn off axis
    plt.savefig(result_image_path, bbox_inches='tight', pad_inches=0)  # Save the figure without extra white space
    plt.close()

    print(f"Predicted image saved to: {result_image_path}")
    return result_image_path
