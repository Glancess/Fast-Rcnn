from torchvision.datasets import VOCDetection

test_dataset = VOCDetection(root=".", year="2007", image_set="test", download=True)
