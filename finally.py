import os
from ultralytics import YOLO
import subprocess
import torch
import numpy as np
import math
import io
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
from datetime import datetime, timedelta
import cv2
from typing import Union
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 图像转换
tf1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 128))
])

tf2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)),
    transforms.CenterCrop((384, 128))
])


def create_directory(directory: str):
    # 检查目录是否存在
    if not os.path.isdir(directory):
        # 如果目录不存在，则创建它
        os.makedirs(directory)
        print(f"目录 {directory} 已创建。")
    else:
        print(f"目录 {directory} 已存在。")


# 字符分割
tokenizer = SimpleTokenizer()
# GPU上运行模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 图像保存路径
save_elder_path = "./save_files/elder"
save_lost_path = "./save_files/lost"
create_directory("./save_files")
create_directory(save_elder_path)
create_directory(save_lost_path)


# 模型加载
def run_model(model_configs: str):
    # config_file = './logs/RSTPReid/20240710_113514_baseline/configs.yaml'
    config_file = model_configs
    args = load_train_configs(config_file)
    args.training = False

    # 加载模型
    model = build_model(args, 3701)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)

    return model


# 文本向量化
def run_texts(texts: Union[tuple, list], model: torch.nn.Module):
    texts_feats = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            # 待寻找目标特征文本
            text = text.strip()

            # 读文本
            text_tensor = tokenize(text, tokenizer, 77, True).unsqueeze(0).to(device)

            # 文本转特征向量
            text_feats = model.encode_text(text_tensor)

            # 转化特征向量->模为1
            text_feats = F.normalize(text_feats, p=2, dim=1)

            texts_feats.append(text_feats)

    return texts_feats


def image_read(image):
    """
    :param image: 二进制图像数据或者图像路径
    :return: Image图像对象，Numpy图像数据
    """
    # 如果是文件路径，则读文件路径
    if isinstance(image, str):
        imgs_path = image.strip()
        # 读图片
        image = read_image(imgs_path)

    # 如果图片是二进制数据，则读二进制
    elif isinstance(image, (bytes, bytearray)):
        # 读图片
        image = Image.open(io.BytesIO(image)).convert("RGB")

    else:
        exit(-1)

    # 转换为numpy数据
    image_data = np.array(image)

    return image, image_data


# 图像向量化
def run_img(image: Image, model: torch.nn.Module, transform: transforms):
    """
    :param image: Image图像对象
    :param model: 深度学习模型
    :param transform: 数据处理方法
    :return: 图像特征向量
    """
    model.eval()
    with torch.no_grad():
        imgs_tensor = transform(image).unsqueeze(0).to(device)
        # 图像转特征向量
        img_feats = model.encode_image(imgs_tensor)

        # 转化特征向量->模为1
        img_feats = F.normalize(img_feats, p=2, dim=1)  # image features

        return img_feats


def run_similarity(first_feats: torch.Tensor, second_feats: torch.Tensor):
    # 计算余弦相似度
    similarity = first_feats @ second_feats.t()

    return similarity.item()


def resize_and_pad_image(image, target_height=350, target_width=200, pad_color=0):
    # 读取图像的尺寸
    orig_height, orig_width = image.shape[0], image.shape[1]

    # 计算缩放比例
    ratio = min((target_width - 80) / orig_width, (target_height - 80) / orig_height)
    # ratio = min(target_width / orig_width, target_height / orig_height)

    # 计算新的尺寸
    new_width = int(orig_width * ratio)
    new_height = int(orig_height * ratio)

    # 对图像进行缩放,cv2.INTER_AREA适用于缩小图像时保持较好的图像质量
    resized_image = cv2.resize(image, (new_width, new_height))

    # return resized_image

    # 计算需要填充的宽度和高度
    top, bottom = (target_height - new_height) // 2, (target_height - new_height) - ((target_height - new_height) // 2)
    left, right = (target_width - new_width) // 2, (target_width - new_width) - ((target_width - new_width) // 2)

    # 创建一个新的全黑背景图像
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    padded_image[:] = pad_color  # 设置填充颜色，默认为黑色,白色则pad_color=[255, 255, 255]

    # 将缩放后的图像放置在中心位置
    padded_image[top:top + new_height, left:left + new_width] = resized_image

    return padded_image


def calculate_lost(time, weather, stay, slow):
    problog_string = f'''
    % Define the probabilities for each condition based on the given sample
    {time}::time.
    {weather}::weather.
    {stay}::stay.
    {slow}::slow.

    % Define 32 rules with their respective prior probabilities
    0::slow.
0::stay. 
0::weather.
0::time. 

0.3::time :- stay.
0.15::weather :- stay. 

0.28::time :- slow, stay. 
0.18::weather :- stay, slow. 
0.3::time :- weather, stay. 

0.12::weather :- time, stay. 

0.37::time :- weather, stay, slow. 
0.15::weather :- time, stay, slow. 
0.37::slow :- weather, stay, time. 
0.30::time :- slow. 

0.27::time :- weather, slow. 



0.70::stay :- weather. 
0.15::time :- weather. 
0.73::slow :- weather. 
0.80::stay :- weather, time. 
0.73::slow :- time. 
0.83::slow :- weather, time. 
0.73::stay :- time. 


0.90::e :- slow, stay, weather, time. 
0.82::e :- slow, stay, weather, \+time. 
0.84::e :- slow, stay, \+weather, time.
0.73::e :- slow, stay, \+weather, \+time. 
0.82::e :- slow, \+stay, weather, time.
0.77::e :- slow, \+stay, weather, \+time. 
0.78::e :- slow, \+stay, \+weather, time. 
0.3::e :- slow, \+stay, \+weather, \+time. 
0.86::e :- \+slow, stay, weather, time. 
0.79::e :- \+slow, stay, weather, \+time.
0.7::e :- \+slow, stay, \+weather, time. 
0.3::e :- \+slow, stay, \+weather, \+time. 
0.80::e :- \+slow, \+stay, weather, time. 
0.65::e :- \+slow, \+stay, weather, \+time. 
0.60::e :- \+slow, \+stay, \+weather, time. 
0.01::e :- \+slow, \+stay, \+weather, \+time. 

query(e).

    '''
    with open("lost.pl", "w") as file:
        file.write(problog_string)
    # 执行命令并获取结果
    result = subprocess.run("problog lost.pl", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 检查命令是否成功执行
    if result.returncode == 0:
        # 打印标准输出
        # print("标准输出:")
        # print(result.stdout)
        return float(result.stdout.split(":")[-1].strip())
    else:
        # 打印错误信息
        print("发生错误:")
        print(result.stderr)


class LostElder:

    def __init__(self, config_file='./logs/RSTPReid/20240710_113514_baseline/configs.yaml'):

        # 加载模型
        self.model = run_model(config_file)
        # 加载匹配文本和文本特征
        self.texts, self.feats = self.features()
        # 加载老人数量
        self.elders_number = 0
        # 加载老人列表
        self.elders_list = []

        # 加载个体检测
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def features(self):
        """
        :return: 文本，文本向量
        """

        all_texts = [
            "Person",
            "Dog",
            "Cat",

            "elderly person",
            "young person",



            "Night",
            "Not night",

            "a person with a backpack",
            "a person without a backpack"

            # "Rain",
            # "Not rain",
            # "Daytime",
            # "Not daytime",
            # "This elderly person may have gotten lost.",
            # "This elderly person may not have gotten lost.",
            # "This elderly person got lost.",
            # "This elderly person did not get lost.",

            "This elderly person is pointing fingers at the air.",
            "This elderly person behaves normally.",
            "This elderly person behaves abnormally.",
            "This elderly person has Medical Adhesive Plasters wrapped around his/her body.",
            "This elderly person doesn't have any Medical Adhesive Plasters wrapped around his/her body.",
            "This elderly person is wearing a blue and white striped medical gown.",
            "This elderly person is not wearing a blue and white striped medical gown.",
            "This elderly person wears a wristband or watch on his/her hand.",
            "This elderly person doesn't wear a wristband or watch on his/her hand.",
            "This elderly person is wearing short sleeved shirt.",
            "This elderly person is not wearing short sleeved shirt.",
            # "This elderly person is wearing long sleeves",
            # "This elderly person is not wearing long sleeves",
            "This elderly person is wearing half pants.",
            "This elderly person is not wearing half pants.",
            "This elderly person is wearing shorts.",
            "This elderly person is not wearing shorts.",
            "This elderly person is wearing pants.",
            # "This elderly person is wearing trousers.",
            # "This elderly person is not wearing trousers.",
            "This elderly person is wearing slippers.",
            "This elderly person is not wearing slippers.",
            "This elderly person is sitting.",
            "This elderly person is walking.",
            "This elderly person is standing."
        ]

        all_feats = run_texts(all_texts, self.model)

        return all_texts, all_feats

        # face = "A elderly person with gray or white hair and facial wrinkles."
        # body = "A elderly person with a hunched back and slow movements."
        # other = "There is a elderly person holding a walking stick in hand."
        # clothes = "There is a elderly person wearing plain clothes."
        # person = ("Elder", "Child", "Dog", "Cat", face, body, other, clothes)

        # face_feats = run_text(face, self.model)
        # body_feats = run_text(body, self.model)
        # other_feats = run_text(other, self.model)
        # clothes_feats = run_text(clothes, self.model)

        # return [person_feats, face_feats, body_feats, other_feats, clothes_feats]

    def check_image_information(self, image_path, confidence_threshold=0.3):
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Invalid image path or unable to read image")

        image_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        model = YOLO("yolo11s.pt")

        results = model(image_data)

        person_images_info = []

        display_image = image_data.copy()

        for result in results:

            boxes = result.boxes.xyxy.cpu().numpy()

            classes = result.boxes.cls.cpu().numpy()

            confidences = result.boxes.conf.cpu().numpy()

            for box, label, conf in zip(boxes, classes, confidences):

                hi = box[2]

                # print("box", box)
                #
                # print("label", label)
                #
                # print("conf", conf)

                if int(label) == 0 and conf >= confidence_threshold:
                    x1, y1, x2, y2 = box.astype(int)

                    center_x = (x1 + x2) // 2

                    center_y = (y1 + y2) // 2

                    person_img = Image.fromarray(image_data[y1:y2, x1:x2, :], 'RGB')

                    img_feat = run_img(person_img, self.model, tf2)

                    person_images_info.append(((person_img, img_feat), (center_x, center_y), hi))

                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cv2.putText(display_image, f"Person: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)

        cv2.imwrite("Detected_Persons.jpg", cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))

        return person_images_info

    def check_if_person(self, image):
        """
        :param image: Image图像对象
        :return: 是否为人，图像特征向量
        """

        img_feat = run_img(image, self.model, tf1)

        # 遍历特征并计算相似度分数
        scores = []
        for feat in self.feats[:3]:
            similarity_score = run_similarity(feat, img_feat)
            scores.append(similarity_score)

        # 放大差异
        scores = torch.tensor(scores) * 100
        # 差异概率归一
        scores = F.softmax(scores, dim=-1)
        # 概率条件化
        person_if = True if scores[0] >= 0.5 else False

        print(f"人：{scores}")


        return person_if, img_feat

    def check_if_elder(self, img_feat):
        """
        :param image: 输入图像特征向量
        :return: 是否为老人
        """

        # 遍历特征并计算相似度分数
        scores = []
        for feat in self.feats[3:5]:
            similarity_score = run_similarity(feat, img_feat)
            scores.append(similarity_score)

        # 放大差异
        scores = torch.tensor(scores) * 100
        # 差异概率归一
        scores = F.softmax(scores, dim=-1)
        # 概率条件化
        elder_if = True if scores[0] >= 0.70 else False



        print(f"老人：{scores}")

        return elder_if

    def check_if_appear(self, img_feat):

        # 计算所有老人的相似度分数
        scores = []
        for elder_info in self.elders_list:
            elder_similarity = []

            # 计算一个老人的所有图像信息的相似度分数
            for elder_feat in elder_info["Image_feat"]:
                similarity_score = run_similarity(img_feat, elder_feat)
                elder_similarity.append(similarity_score)

            # 取平均分数
            scores.append(sum(elder_similarity) / len(elder_similarity))

        # 放大差异
        scores = torch.tensor(scores) * 100
        # print(scores, scores.shape)

        # 获取最大值索引
        max_index = torch.argmax(scores, dim=-1)

        # 是否出现过
        appear_if = True if scores[max_index] >= 66 else False

        if appear_if:
            print(f"该老人为出现过的老人的可能性得分：{scores},老人编号索引为{max_index}.")
        else:
            print(f"该老人未出现过.")

        return appear_if, max_index

    def check_if_night(self, image: Image):

        img_feat = run_img(image, self.model, tf2)
        # 遍历特征并计算相似度分数
        scores = []
        for feat in self.feats[6:8]:
            similarity_score = run_similarity(feat, img_feat)
            scores.append(similarity_score)

        # 放大差异
        scores = torch.tensor(scores) * 100
        # 差异概率归一
        scores = F.softmax(scores, dim=-1)
        # # 概率条件化
        # night_if = True if scores[0] > scores[1] else False

        # 夜晚概率
        # night_p = scores[0]
        #
        # print(f"夜晚的概率:{night_p}")
        #
        # return night_p, img_feat

    # def check_if_bag(self, img_feat):
    #     """
    #     :param image: 输入图像特征向量
    #     :return: 是否为老人
    #     """
    #
    #     # 遍历特征并计算相似度分数
    #     scores = []
    #     for feat in self.feats[8:10]:
    #         similarity_score = run_similarity(feat, img_feat)
    #         scores.append(similarity_score)
    #
    #     # 放大差异
    #     scores = torch.tensor(scores) * 100
    #     # 差异概率归一
    #     scores = F.softmax(scores, dim=-1)
    #     # 概率条件化
    #     bag_if = True if scores[0] >= 0.50 else False
    #
    #     # 如果不是老人,输出概率值
    #
    #     print(f"背包：{scores}")
    #
    #     return bag_if

    def autorun(self, image, latitude, longitude):
        lost_pr=[]
        # 检测个体信息
        person_images_info = self.check_image_information(image)
        # 计算人数
        people_number = len(person_images_info)


        if people_number != 0:
            print("human")

            for (image_data, img_feat), (loc_x, loc_y), hi in person_images_info:

                # 检测是否为老人
                elder_if = self.check_if_elder(img_feat)
                # bag_if = self.check_if_bag(img_feat)

                if elder_if:

                    if self.elders_number > 0:
                        # 判断是否出现
                        elder_appear_if, elder_number = self.check_if_appear(img_feat)
                    else:
                        elder_appear_if, elder_number = False, 0

                    # 获取当前时间
                    time_now = datetime.now()
                    formatted_now = time_now.strftime("%Y-%m-%d %H:%M:%S")

                    # 如果老人出现过
                    if elder_appear_if:
                        now = datetime.now()
                        timestamp = now.timestamp()
                        # 保存图像文件
                        image_data.save(
                            os.path.join(save_elder_path,
                                         f"elder{elder_number}_{len(self.elders_list[elder_number])}_{timestamp}.png"),
                            "PNG")
                        # 在老人列表里的获取该老人信息字典
                        this_elder_info_dict = self.elders_list[elder_number]

                        last_loc_x_y = this_elder_info_dict["Loc_x_y"][-1]  # 读取上一时刻的位置
                        time_last = datetime.strptime(this_elder_info_dict["Time"][-1],
                                                      "%Y-%m-%d %H:%M:%S")  # 读取上一时刻的时间
                        diff_time_now_last = (time_now - time_last).seconds  # 获得时间差
                        if diff_time_now_last != 0:
                            vx = (loc_x - last_loc_x_y[0]) / diff_time_now_last
                            vy = (loc_y - last_loc_x_y[1]) / diff_time_now_last
                        else:
                            # 可以设置vx和vy为0或其他默认值
                            vx = 0
                            vy = 0
                        v = (vx ** 2 + vy ** 2) ** 0.5  # 计算速度和
                        v = 1.7*v/hi

                        time_first = datetime.strptime(this_elder_info_dict["Time"][0],
                                                       "%Y-%m-%d %H:%M:%S")  # 读取最先时刻的时间
                        diff_time_now_first = (time_now - time_first).seconds  # 获得时间差
                        diff_time_now_first = diff_time_now_first/60
                        linger_fun = lambda x: 1/(1 + 5.6* math.exp(-0.05 * x)) # 逗留概率函数
                        linger_p = linger_fun(diff_time_now_first)  # 逗留概率
                        print(f"逗留概率：{linger_p}")

                        # 转向次数->徘徊方向->徘徊概率
                        # hesitate_num = this_elder_info_dict["Hesitate"]["Hesitate_num"][-1]
                        # last_vx = this_elder_info_dict["Vx"][-1]
                        # if last_vx < 0 < vx or vx < 0 < last_vx:
                        #     hesitate_num += 1
                        # if hesitate_num > 1:
                        #     hesitate_fun = lambda x: -math.exp(-(x / 3.3) ** 2) + 1  # 徘徊概率函数
                        #     hesitate_p = hesitate_fun(hesitate_num)
                        # else:
                        #     hesitate_p = 0
                        print(f"速度{v}")
                        slow_fun = lambda x: 2.4*0.0006**x  # 行走缓慢，走走停停，概率函数
                        if v>0.2:
                            slow_p = min(1,max(0,slow_fun(v)))
                        else:
                            slow_p = 0
                        print(f"走得慢概率：{slow_p}")

                        single_p = 1 / people_number  # 独自概率

                        #夜晚概率
                        current_time = datetime.now()

                        # 如果你需要一个从当天午夜开始的秒数表示（仍然是一个整数或浮点数，但更常用于计算时间差）
                        seconds_since_midnight = (
                                current_time - current_time.replace(hour=0, minute=0, second=0,
                                                                    microsecond=0)).total_seconds()

                        t = seconds_since_midnight / 3600
                        night_fun = lambda x: 1/(1 + 23387300 * math.exp(-0.81 * x))
                        if 0<t<5:
                            night_p = 1
                        elif 5<t<18:
                            night_p = 0
                        else:
                            night_p = night_fun(t)
                        print(f"夜晚概率：{night_p}")

                        #恶劣天气概率
                        # badweather_p = input("输入恶劣天气概率：")
                        badweather_p = 0

                        # 计算当前迷失概率
                        lost_p = calculate_lost(weather=badweather_p, time=night_p,
                                                stay=linger_p, slow=slow_p,
                                               )

                        # 添加该老人信息
                        this_elder_info_dict["Image_feat"].append(img_feat)
                        this_elder_info_dict["Loc_x_y"].append((loc_x, loc_y))
                        this_elder_info_dict["Location"].append((latitude, longitude))
                        this_elder_info_dict["Time"].append(formatted_now)
                        this_elder_info_dict["V"].append(v)
                        this_elder_info_dict["Vx"].append(vx)
                        this_elder_info_dict["Vy"].append(vy)
                        this_elder_info_dict["Linger_p"].append(linger_p)
                        # this_elder_info_dict["Hesitate"]["Hesitate_num"].append(hesitate_num)
                        # this_elder_info_dict["Hesitate"]["Hesitate_p"].append(hesitate_p)
                        this_elder_info_dict["Night_p"].append(night_p)
                        this_elder_info_dict["Single_p"].append(single_p)
                        this_elder_info_dict["Slow_p"].append(slow_p)
                        this_elder_info_dict["Lost_p"].append(lost_p)
                        this_elder_info_dict["Badweather_p"].append(badweather_p)
                        this_elder_info_dict["Hi"].append(hi)

                        # print(this_elder_info_dict)
                        print(f"迷失概率：{lost_p}")
                        if lost_p > 0.5:
                            print(f"老人可能发生迷失，概率为{lost_p},坐标为:({latitude},{longitude})")
                            image_data.save(os.path.join(save_lost_path,
                                                         f"elder{elder_number}_{len(self.elders_list[elder_number])}.png"),
                                            "PNG")
                        lost_pr.append(lost_p)
                    else:
                        now = datetime.now()
                        timestamp = now.timestamp()
                        self.elders_list.append(
                            {"Image_feat": [img_feat],
                             "Loc_x_y": [(loc_x, loc_y)],
                             "Location": [(latitude, longitude)],
                             "Time": [formatted_now],
                             "Hi": [hi],
                             "V": [0],
                             "Vx": [0],
                             "Vy": [0],
                             "Linger_p": [0],
                             "Hesitate": {"Hesitate_num": [0],
                                          "Hesitate_p": [0]},
                             "Night_p": [0],
                             "Single_p": [0],
                             "Slow_p": [0],
                             "Lost_p": [0],
                             "Badweather_p": [0]
                             })
                        # 保存图像文件
                        image_data.save(os.path.join(save_elder_path, f"elder{self.elders_number}_0_{timestamp}.png"), "PNG")
                        self.elders_number += 1
            return lost_pr

                        # return 0

        return -1
    def check_if_linger(self, index):

        # 获取对应老人信息
        elder_info = self.elders_list[index]

        # 获取第一次记录时间
        time1 = datetime.strptime(elder_info[0][2], "%Y-%m-%d %H:%M:%S")
        # 获取最后一次记录时间
        time2 = datetime.strptime(elder_info[-1][2], "%Y-%m-%d %H:%M:%S")
        # 计算两个时间之间的差异
        diff = time2 - time1

        # 是否逗留(逗留时间超过阈值)
        linger_if = True if diff >= timedelta(seconds=30) else False

        if linger_if:
            print(f"逗留老人编号为：number{index}")

        return linger_if

    def check_if_hesitate(self):
        pass

    def check_if_abnormal(self, img_feat):
        # 记录各姿态得分
        scores = []
        for action_feat in self.feats[9:]:
            similarity = run_similarity(img_feat, action_feat)
            scores.append(similarity)

        # 放大差异
        scores = torch.tensor(scores) * 100
        # 获取最大值索引
        # max_index = torch.argmax(scores, dim=-1)
        # 差异概率归一
        # scores = F.softmax(scores, dim=-1)
        # 概率条件化
        # elder_if = True if scores[max_index] >= 0.98 else False

        for text, score in zip(self.texts[8:], scores):
            print(f"{text}:{score}")


def check_images_in_folder(folder_path):
    image_extensions = ('.jpg', '.jpeg', '.png')

    image_paths = []

    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"文件夹 '{folder_path}' 不存在")

    for file_path in folder.rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            image_paths.append(str(file_path))

    return image_paths


def delete_image(image_path):
    try:
        os.remove(image_path)
        print(f"已删除文件: {image_path}")
    except Exception as e:
        print(f"删除文件时出错: {str(e)}")

if __name__ == '__main__':
    check = LostElder()



    folder_path = "E:\python_projects\img_receive"
    while True:

        image_paths = check_images_in_folder(folder_path)

        if image_paths:
            print("\n找到以下图片文件:")
            for path in image_paths:
                print(path)
                check.autorun(path, "GPS1", "GPS2")
                delete_image(path)
                # time.sleep(0.5)

        else:
            print("\n未找到图片文件，等待1秒后重试...")
            time.sleep(1)





