import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import os.path as op
import torch.nn.functional as F
from model.build import build_model
from utils.iotools import load_train_configs, read_image
import matplotlib.pyplot as plt
from PIL import Image
from datasets.bases import tokenize
from utils.checkpoint import Checkpointer
from utils.simple_tokenizer import SimpleTokenizer
from torchvision import transforms

def get_one_query_caption_and_result_by_id(idx, indices, qids, gids, captions, img_paths):
    query_caption = captions[idx]
    query_id = qids[idx]
    image_paths = [img_paths[j] for j in indices[idx]]
    image_ids = gids[indices[idx]]
    # gt_image_path = gt_img_paths[idx]
    # return query_id, image_ids, query_caption, image_paths, gt_image_path
    return query_id, image_ids, query_caption, image_paths


def plot_retrieval_images(query_id, image_ids, query_caption, image_paths, fname=None):
    print(query_id)
    print(image_ids)
    print(query_caption)

    col = len(image_paths)

    for i in range(col):
        # plt.subplot(1, col + 1, i + 2)
        plt.subplot(1, col + 1, i + 1)
        img = Image.open(image_paths[i])
        img = img.resize((128, 256))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    plt.show()






if __name__ == '__main__':
    config_file = './logs/RSTPReid/20240710_113514_baseline/configs.yaml'
    args = load_train_configs(config_file)
    args.training = False

    device = torch.device("cuda")
    # 加载模型
    model = build_model(args,3701)
    # model.to(device)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    tokenizer = SimpleTokenizer()
    model.to(device)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((384, 128))
    ])

    model.eval()
    with torch.no_grad():
        test_txts = input("请输入待寻找目标特征:").strip()
        # test_txts = "This a chicken,the chicken is playing basketball."
        while True:
            test_imgs_path = input("请输入图像文件路径:").strip()
            # test_imgs_path = "xiaoheizi.png"
            if test_imgs_path == "":
                break

            # 读图片、读文本
            test_imgs_tensor = tf(read_image(test_imgs_path)).unsqueeze(0).to(device)
            test_txts_tensor = tokenize(test_txts,tokenizer,77,True).unsqueeze(0).to(device)

            # print(test_imgs)
            # print(test_txts)

            # 图像和文本转特征向量
            txt_feats = model.encode_text(test_txts_tensor)
            img_feats = model.encode_image(test_imgs_tensor)

            # 计算余弦相似度
            txt_feats = F.normalize(txt_feats, p=2, dim=1)  # text features
            img_feats = F.normalize(img_feats, p=2, dim=1)  # image features
            similarity = txt_feats @ img_feats.t()

            print(similarity.shape)
            print(similarity)


