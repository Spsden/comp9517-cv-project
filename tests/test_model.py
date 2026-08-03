import torch

from inat_project.models import create_resnet18


def test_resnet18_output_shape():
    model = create_resnet18(num_classes=500, pretrained=False)
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 500)

