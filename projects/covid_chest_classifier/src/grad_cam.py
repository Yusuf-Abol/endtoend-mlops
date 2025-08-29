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

        # Only use full backward hook (PyTorch 2.x)
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[:, target_class].sum()
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

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
    if org_img.ndim == 2:
        org_img = np.stack([org_img]*3, axis=-1)
    overlayed = heatmap * alpha + org_img
    overlayed = overlayed / np.max(overlayed)
    return overlayed

# -----------------------------
# Display and/or save Grad-CAM
# -----------------------------
def save_and_display_gradcam(original, overlay, out_path, fname, class_name="Unknown", show_inline=True):
    overlay = np.clip(overlay, 0, 1)
    original = np.clip(original, 0, 1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original, cmap="gray")
    axs[0].set_title(class_name)                 # show actual class
    axs[0].axis("off")

    axs[1].imshow(original, cmap="gray")
    axs[1].imshow(overlay, cmap="jet", alpha=0.4)
    axs[1].set_title(f"{class_name} - Grad-CAM")
    axs[1].axis("off")

    os.makedirs(out_path, exist_ok=True)
    save_path = os.path.join(out_path, fname)
    plt.savefig(save_path, bbox_inches="tight")
    if show_inline:
        plt.show()
    plt.close(fig)
    print(f"[INFO] Saved Grad-CAM comparison to {save_path}")

# -----------------------------
# Generate Grad-CAM from DataLoader
# -----------------------------
def generate_gradcam(model, dataloader, classes, save_dir="gradcam_outputs", num_images=5, device="cpu"):
    model = model.to(device).eval()
    processed = 0

    target_layer = model.layer4[1].conv2  # last conv layer for ResNet18
    gradcam = GradCAM(model, target_layer)

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        for i in range(imgs.size(0)):
            img_tensor = imgs[i].unsqueeze(0)
            img_tensor.requires_grad = True
            label_idx = labels[i].item()
            class_name = classes[label_idx]

            # Grad-CAM
            cam = gradcam.generate(img_tensor, target_class=label_idx)

            # Overlay
            img_np = imgs[i].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            overlay = apply_colormap_on_image(img_np, cam)

            # Save and display with class name
            fname = f"{processed}_{class_name}.png"
            save_and_display_gradcam(img_np, overlay, save_dir, fname, class_name=class_name)

            processed += 1
            if processed >= num_images:
                return
