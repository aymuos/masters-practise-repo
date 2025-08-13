# model download
mobilenet = models.mobilenet_v2(pretrained=False)
mobilenet.load_state_dict(torch.load('/opt/spark/data/pretrained_models/mobilenet_v2-b0353104.pth'))

# data download
mnist_train = datasets.MNIST(root="./data", train=True, download=True) mnist_test = datasets.MNIST(root="./data", train=False, download=True)
