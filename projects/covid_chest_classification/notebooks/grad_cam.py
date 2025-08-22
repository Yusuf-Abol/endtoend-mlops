import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# -----------------------------
# Grad-CAM class
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # hook for forward pass
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        # hook for backward pass
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        # forward
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # backward
        self.model.zero_grad()
        loss = output[:, target_class].sum()
        loss.backward()

        # Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# -----------------------------
# Helper to overlay heatmap
# -----------------------------
def apply_colormap_on_image(org_img, activation_map, alpha=0.4):
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    org_img = np.float32(org_img) / 255
    if org_img.ndim == 2:  # grayscale -> RGB
        org_img = np.stack([org_img]*3, axis=-1)
    overlayed = heatmap * alpha + org_img
    overlayed = overlayed / np.max(overlayed)
    return overlayed


# -----------------------------
# Save side-by-side result
# -----------------------------
def save_gradcam_result(original, overlay, out_path, fname):
    # clip values
    overlay = np.clip(overlay, 0, 1)
    original = np.clip(original, 0, 1)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(original, cmap="gray")
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(original, cmap="gray")
    axs[1].imshow(overlay, cmap="jet", alpha=0.4)
    axs[1].set_title("Grad-CAM")
    axs[1].axis("off")

    os.makedirs(out_path, exist_ok=True)
    save_path = os.path.join(out_path, fname)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved Grad-CAM comparison to {save_path}")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    from torchvision import models, transforms
    from PIL import Image

    device = torch.device("cpu")
    model = models.resnet18(pretrained=True).to(device).eval()

    # pick last conv layer
    target_layer = model.layer4[1].conv2

    gradcam = GradCAM(model, target_layer)

    # transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # load image
    img_path = "sample_xray.jpg"   # <-- replace with your chest X-ray path
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Grad-CAM
    cam = gradcam.generate(input_tensor)

    # overlay
    overlay = apply_colormap_on_image(np.array(img), cam)

    # save
    save_gradcam_result(np.array(img), overlay, "gradcam_outputs", "gradcam_example.png")
