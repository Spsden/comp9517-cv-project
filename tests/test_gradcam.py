import torch

from inat_project.gradcam import GradCAM
from inat_project.models import create_resnet18


def test_gradcam_shape_and_range():
    model = create_resnet18(num_classes=8, pretrained=False)
    sample = torch.randn(1, 3, 64, 64)

    with GradCAM(model, model.layer4[-1]) as gradcam:
        cam, logits = gradcam(sample)

    assert cam.shape == (1, 64, 64)
    assert logits.shape == (1, 8)
    assert torch.isfinite(cam).all()
    assert cam.min() >= 0
    assert cam.max() <= 1

