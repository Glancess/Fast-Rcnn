from torchvision.datasets import VOCDetection

if __name__ == "__main__":
    # 只是 Dataset 小实验；import 时不能自动下载数据。
    test_dataset = VOCDetection(root="./data/test", year="2007", image_set="test", download=False)
    print("test images:", len(test_dataset))
