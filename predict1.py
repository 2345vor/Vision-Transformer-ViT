import os
import json
import argparse
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import CAFIA_Transformer
# 获取预测结果
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def predict():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img_path = "./image/bird.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    #img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default='./output', type=str)
    parser.add_argument("--vit_model", default='./Vit_weights/imagenet21k+imagenet2012_ViT-B_16-224.pth', type=str)
    parser.add_argument("--image_size", type=int, default=224, help="input image size", choices=[224, 384])
    parser.add_argument("--num-classes", type=int, default=10, help="number of classes in dataset")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--mlp_dim", type=int, default=3072)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--attn_dropout_rate", type=float, default=0.0)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    args = parser.parse_args()


    model = CAFIA_Transformer(args)
    model.to(device)  #has_logits=False要和训练时（train.py）设置的一样
    # load model weights
    model_weight_path = "./output/0.9912199974060059.pt"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    if predict[predict_cla].numpy() <0.5:
        print_res = "class: {}   prob: {:.3}".format('no match',
                                                     predict[predict_cla].numpy())
    else:
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    predict()
