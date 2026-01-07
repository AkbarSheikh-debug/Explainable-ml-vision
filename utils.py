import os
import io
import random
import time
import numpy as np
import copy
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import seaborn as sns
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
import warnings
warnings.filterwarnings('ignore')

class ExplainableML:
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = ['ants', 'bees']
        self.transform = None
        self.features_blobs = []
        
    def initialize_model(self, model_name='resnext101', num_classes=2, feature_extract=True):
        """Initialize the model with given architecture"""
        model = None
        
        if model_name == "resnext101":
            model = models.resnext101_32x8d(pretrained=True)
            self.set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            input_size = (224, 224)
            self.finalconv_name = "layer4"
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            self.set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            input_size = (224, 224)
            self.finalconv_name = "layer4"
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=True)
            self.set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            input_size = (224, 224)
            self.finalconv_name = "features"
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        return model, input_size
    
    def set_parameter_requires_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
    
    def get_target_layer(self, model):
        """Get the target layer for Grad-CAM visualization"""
        if not hasattr(self, 'finalconv_name'):
            raise ValueError("finalconv_name not set. Initialize model first.")
        
        target_layer = None
        
        # Try direct access
        if hasattr(model, self.finalconv_name):
            target_layer = getattr(model, self.finalconv_name)
        # Try _modules access
        elif hasattr(model, '_modules') and self.finalconv_name in model._modules:
            target_layer = model._modules[self.finalconv_name]
        else:
            # Search through named modules
            for name, module in model.named_modules():
                if name == self.finalconv_name or name.endswith('.' + self.finalconv_name):
                    target_layer = module
                    break
        
        if target_layer is None:
            raise ValueError(f"Could not find target layer '{self.finalconv_name}' in model")
        
        # Ensure gradients are enabled for the target layer
        for param in target_layer.parameters():
            param.requires_grad = True
        
        return target_layer
    
    def create_data_transforms(self, input_size=(224, 224)):
        """Create data transformations for training and inference"""
        data_transforms = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "val": transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "test": transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        return data_transforms
    
    def train_model(self, model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
        """Train the model"""
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs-1}')
            print('-' * 10)
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                
                running_loss = 0.0
                running_corrects = 0
                
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accs.append(epoch_acc.cpu().numpy())
                else:
                    val_losses.append(epoch_loss)
                    val_accs.append(epoch_acc.cpu().numpy())
            
            # Step the learning rate scheduler after each epoch
            if scheduler is not None:
                scheduler.step()
            
            print()
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        
        model.load_state_dict(best_model_wts)
        return model, train_losses, val_losses, train_accs, val_accs
    
    def generate_grad_cam(self, image_path, model, target_layer, target_class=None):
        """Generate Grad-CAM visualization"""
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(img).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True
        
        # Get prediction first
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            pred_class = torch.argmax(probs).item()
        
        # Check if target_layer is valid
        if target_layer is None:
            raise ValueError("target_layer cannot be None. Check if model architecture matches expected layer name.")
        
        # Try using pytorch-grad-cam library first
        grayscale_cam = None
        cam = None
        cam_initialized = False
        try:
            # Initialize Grad-CAM: handle different library versions and CPU-only setups
            try:
                cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(self.device.type == 'cuda'))
                cam_initialized = True
            except TypeError:
                try:
                    cam = GradCAM(model=model, target_layers=[target_layer], device=self.device)
                    cam_initialized = True
                except TypeError:
                    cam = GradCAM(model=model, target_layers=[target_layer])
                    cam_initialized = True
            
            # Ensure the cam object is properly initialized before use
            if cam is not None and cam_initialized:
                # Try different API versions
                try:
                    # Newer API: use targets parameter
                    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                    targets = [ClassifierOutputTarget(pred_class if target_class is None else target_class)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                    # Mark that cam was successfully used
                    cam_initialized = True
                except (TypeError, AttributeError):
                    try:
                        # Older API: just pass input_tensor
                        grayscale_cam = cam(input_tensor=input_tensor)
                        cam_initialized = True
                    except Exception:
                        # If all library calls fail, use manual implementation
                        grayscale_cam = None
                        cam_initialized = False
        except Exception:
            # Library initialization or call failed, use manual implementation
            grayscale_cam = None
            cam_initialized = False
        finally:
            # Ensure proper cleanup of GradCAM object to prevent AttributeError on destruction
            if cam is not None:
                try:
                    # Only try to release if the object was successfully initialized and used
                    if cam_initialized:
                        # Check if activations_and_grads exists before accessing
                        if hasattr(cam, 'activations_and_grads'):
                            if cam.activations_and_grads is not None:
                                try:
                                    cam.activations_and_grads.release()
                                except (AttributeError, RuntimeError):
                                    # Ignore cleanup errors
                                    pass
                    # Set to None to help with garbage collection
                    cam = None
                except Exception:
                    # Ignore any cleanup errors - object will be garbage collected naturally
                    cam = None
                    pass
        
        # Fallback to manual Grad-CAM implementation
        if grayscale_cam is None:
            activations = []
            gradients = []

            def forward_hook(module, inp, out):
                activations.append(out.clone())

            def backward_hook(module, grad_in, grad_out):
                if grad_out[0] is not None:
                    gradients.append(grad_out[0].clone())

            # Register hooks
            handle_forward = target_layer.register_forward_hook(forward_hook)
            
            # Use register_full_backward_hook for better compatibility
            def full_backward_hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    gradients.append(grad_output[0].clone())
            
            if hasattr(target_layer, 'register_full_backward_hook'):
                handle_backward = target_layer.register_full_backward_hook(full_backward_hook)
            else:
                handle_backward = target_layer.register_backward_hook(backward_hook)

            try:
                # Ensure model is in eval mode but gradients enabled
                model.eval()
                
                # Forward pass
                model.zero_grad()
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                chosen = target_class if target_class is not None else torch.argmax(probs).item()

                # Backward pass on the score for the chosen class
                score = output[0, chosen]
                score.backward(retain_graph=False)

                # Check if we got activations and gradients
                if len(activations) == 0:
                    raise RuntimeError('Failed to capture activations - forward hook did not fire')
                
                if len(gradients) == 0:
                    raise RuntimeError('Failed to capture gradients - backward hook did not fire or gradients are None')

                activation = activations[-1]
                gradient = gradients[-1]

                # Handle case where gradient might be None
                if gradient is None:
                    raise RuntimeError('Gradient is None - model may not support gradient computation')

                # Global-average-pool the gradients
                # Handle different tensor shapes
                if len(gradient.shape) == 4:
                    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
                else:
                    weights = gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)

                # Weighted combination of activations
                if len(activation.shape) == 4:
                    cam_map = torch.sum(weights * activation, dim=1, keepdim=True)
                else:
                    cam_map = torch.sum(weights * activation, dim=1, keepdim=True)
                
                cam_map = torch.relu(cam_map)

                # Normalize and resize
                cam_map = cam_map.squeeze().cpu().detach().numpy()
                if cam_map.ndim == 0:
                    cam_map = np.array([cam_map])
                cam_map = cam_map - cam_map.min()
                if cam_map.max() != 0:
                    cam_map = cam_map / cam_map.max()
                cam_map = cv2.resize(cam_map, (224, 224))
                grayscale_cam = np.expand_dims(cam_map, axis=0)

            finally:
                # Always remove hooks
                handle_forward.remove()
                handle_backward.remove()
        
        # Prepare visualization
        rgb_img = np.array(img.resize((224, 224))) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
        
        return visualization, pred_class, probs.cpu().numpy()
    
    def generate_lime_explanation(self, image_path, model, num_samples=1000):
        """Generate LIME explanation"""
        explainer = lime_image.LimeImageExplainer()
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img.resize((224, 224))) / 255.0
        
        def batch_predict(images):
            model.eval()
            batch_tensors = []
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            for im in images:
                pil = Image.fromarray((im * 255).astype(np.uint8)).convert('RGB')
                tensor = transforms.ToTensor()(pil)
                tensor = normalize(tensor)
                batch_tensors.append(tensor)

            batch = torch.stack(batch_tensors, dim=0).to(self.device)

            with torch.no_grad():
                logits = model(batch)
                probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()
        
        # Get explanation
        explanation = explainer.explain_instance(
            img_array,
            batch_predict,
            top_labels=2,
            hide_color=0,
            num_samples=num_samples
        )
        
        return explanation, img_array
    
    def generate_shap_explanation(self, image_path, model, background_images=None):
        """Generate SHAP explanation"""
        # Ensure model is in eval mode
        model.eval()
        
        # Create a wrapper model to prevent in-place operations
        # We'll use the model directly but ensure inputs are properly handled
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                # Ensure input is contiguous (but don't clone to preserve gradients)
                if not x.is_contiguous():
                    x = x.contiguous()
                return self.model(x)
        
        wrapped_model = ModelWrapper(model)
        wrapped_model.eval()
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Create input tensor and ensure it's a new tensor (not a view)
        input_tensor = transform(img).unsqueeze(0).to(self.device)
        # Ensure tensor is contiguous and cloned to avoid view issues
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()
        # Clone and detach first, then set requires_grad
        # This ensures we have a fresh tensor that's not a view
        input_tensor = input_tensor.clone().detach()
        input_tensor.requires_grad_(True)
        # Ensure it's still contiguous after setting requires_grad
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()
        
        # Prepare background - ensure it's cloned and contiguous to avoid in-place modification issues
        if background_images is None:
            # Use a smaller, more stable background set
            # Create background from multiple transformed versions of the image
            background_list = []
            for _ in range(5):
                bg_tensor = transform(img).unsqueeze(0).to(self.device)
                # Ensure each tensor is contiguous and cloned before adding to list
                if not bg_tensor.is_contiguous():
                    bg_tensor = bg_tensor.contiguous()
                # Clone to ensure it's not a view
                bg_tensor = bg_tensor.clone().detach()
                background_list.append(bg_tensor)
            # Concatenate and ensure final tensor is contiguous and cloned
            background = torch.cat(background_list, dim=0)
            if not background.is_contiguous():
                background = background.contiguous()
            # Final clone to ensure no views remain
            background = background.clone().detach()
            # Ensure requires_grad is False for background
            background.requires_grad_(False)
        else:
            # Clone background images to avoid in-place modification
            if isinstance(background_images, torch.Tensor):
                if not background_images.is_contiguous():
                    background = background_images.contiguous().clone().detach()
                else:
                    background = background_images.clone().detach()
            else:
                background = background_images
        
        # Try using GradientExplainer first as it's more stable for vision models
        try:
            explainer = shap.GradientExplainer(wrapped_model, background)
            shap_values = explainer.shap_values(input_tensor)
            
            # Convert for visualization - handle different shapes
            shap_numpy = []
            if isinstance(shap_values, list):
                for s in shap_values:
                    if isinstance(s, torch.Tensor):
                        s_np = s.clone().detach().cpu().numpy()
                    else:
                        s_np = np.array(s)
                    
                    # Handle shape conversion: from (batch, channels, height, width) to (height, width, channels)
                    if len(s_np.shape) == 4:
                        s_np = s_np[0]  # Remove batch dimension
                    if len(s_np.shape) == 3:
                        # Convert from (C, H, W) to (H, W, C)
                        s_np = np.transpose(s_np, (1, 2, 0))
                    shap_numpy.append(s_np)
            else:
                if isinstance(shap_values, torch.Tensor):
                    s_np = shap_values.clone().detach().cpu().numpy()
                else:
                    s_np = np.array(shap_values)
                
                # Handle shape conversion
                if len(s_np.shape) == 4:
                    s_np = s_np[0]  # Remove batch dimension
                if len(s_np.shape) == 3:
                    # Convert from (C, H, W) to (H, W, C)
                    s_np = np.transpose(s_np, (1, 2, 0))
                shap_numpy.append(s_np)
            
            # Convert test image for visualization (denormalize)
            test_img_tensor = transform(img)
            test_img_np = test_img_tensor.numpy().transpose((1, 2, 0))
            # Denormalize ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            test_img_np = test_img_np * std + mean
            test_img_np = np.clip(test_img_np, 0, 1)
            test_numpy = np.array([test_img_np])
            
            return shap_numpy, test_numpy
            
        except Exception as e1:
            # Fallback to DeepExplainer if GradientExplainer fails
            try:
                explainer = shap.DeepExplainer(wrapped_model, background)
                shap_values = explainer.shap_values(input_tensor)
                
                # Convert for visualization - handle different shapes
                shap_numpy = []
                if isinstance(shap_values, list):
                    for s in shap_values:
                        if isinstance(s, torch.Tensor):
                            s_np = s.clone().detach().cpu().numpy()
                        else:
                            s_np = np.array(s)
                        
                        # Handle shape conversion: from (batch, channels, height, width) to (height, width, channels)
                        if len(s_np.shape) == 4:
                            s_np = s_np[0]  # Remove batch dimension
                        if len(s_np.shape) == 3:
                            # Convert from (C, H, W) to (H, W, C)
                            s_np = np.transpose(s_np, (1, 2, 0))
                        shap_numpy.append(s_np)
                else:
                    if isinstance(shap_values, torch.Tensor):
                        s_np = shap_values.clone().detach().cpu().numpy()
                    else:
                        s_np = np.array(shap_values)
                    
                    # Handle shape conversion
                    if len(s_np.shape) == 4:
                        s_np = s_np[0]  # Remove batch dimension
                    if len(s_np.shape) == 3:
                        # Convert from (C, H, W) to (H, W, C)
                        s_np = np.transpose(s_np, (1, 2, 0))
                    shap_numpy.append(s_np)
                
                # Convert test image for visualization (denormalize)
                test_img_tensor = transform(img)
                test_img_np = test_img_tensor.numpy().transpose((1, 2, 0))
                # Denormalize ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                test_img_np = test_img_np * std + mean
                test_img_np = np.clip(test_img_np, 0, 1)
                test_numpy = np.array([test_img_np])
                
                return shap_numpy, test_numpy
                
            except Exception as e2:
                import traceback
                error_trace = traceback.format_exc()
                raise RuntimeError(f"SHAP explanation failed with both GradientExplainer and DeepExplainer.\nGradientExplainer error: {str(e1)}\nDeepExplainer error: {str(e2)}\n\nTraceback:\n{error_trace}")
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf
    
    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(train_losses, label='Train Loss')
        axes[0].plot(val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(train_accs, label='Train Accuracy')
        axes[1].plot(val_accs, label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return fig, buf