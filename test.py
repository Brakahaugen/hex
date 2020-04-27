import torch
import torchvision

model = torchvision.models.resnet34()

print(model)
print("printin conv1")
print(model.conv1)
# for layer in model:
#     if type(layer) == torch.nn.Conv2d:
#         print(layer)
