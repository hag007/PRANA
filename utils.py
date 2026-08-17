import torch

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tensor_dtype = "float32"


# Wrapper class to automatically move tensors to the device
class AutoToDevice:
    def __call__(self, *args, **kwargs):
        tensor = torch.tensor(*args, **kwargs)
        return tensor.to(device)


# Create an instance of the wrapper class
auto_to_device = AutoToDevice()
