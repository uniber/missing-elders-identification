from matplotlib import pyplot as plt
import re
import numpy as np

real_epochs = 46
batchs = 1157
train_log_path = "./RSTPReid/20240710_113514_baseline/train_log.txt"

# 记录当前批次数量
cur_count = 0
# 记录每个epoch的每个小批次的所有损失
losses_per_batch = []
# 记录每个epoch的所有损失
losses_per_epoch = []
# 记录每个epoch的验证分数
val_scores_per_epoch = []



with open(train_log_path, mode="r", encoding="utf-8") as f:
    start_bool = False

    # 检查是否为起始位置（训练开始）
    while not start_bool:
        line = f.readline().strip().split(" ")
        start_point = " ".join(line[-2:])
        start_bool = start_point == "start training"

    # 接着逐行读取文件内容
    for line in f:
        line = line.strip()
        # 2024-07-10 13:00:11,158 IRRA.train INFO: Epoch[7] Iteration[1157/1157], loss: 18.2264, sdm_loss: 7.6250, id_loss: 7.8948, mlm_loss: 2.7066, img_acc: 0.6510, txt_acc: 0.3211, mlm_acc: 0.4428, Base Lr: 9.99e-06
        search_mode = re.search(r"(IRRA\.train)|(IRRA\.eval)", line)
        if search_mode:
            if search_mode.group(0) == "IRRA.train":
                line = line.split(",")
                train_logs = line[1].split(" ")
                # ["158","IRRA.train","INFO:","Epoch[7]","Iteration[1157/1157]"]

                # 累加
                cur_count += 1

                # 当长度为5时，才为训练过程数据
                if len(train_logs) == 5:
                    cur_epoch = int(train_logs[-2].split("[")[-1].strip("]"))
                    # ["Epoch[","7]"]->"7"->7

                    if cur_epoch > real_epochs:
                        break

                    train_losses = [float(i.split(" ")[-1]) for i in line[2:6]]
                    """
                    temp_list = []
                    for i in line[2:6]:
                        temp_data = i.split(" ") # list
                        temp_data = temp_data[-1]
                        temp_data = float(temp_data)
                        temp_list.append(temp_data)
                    train_losses = temp_list
                    """
                    #[ loss: 18.2264, sdm_loss: 7.6250, id_loss: 7.8948, mlm_loss: 2.7066]->[18.2264, 7.6250, 7.8948, 2.7066]

                    losses_per_batch.append(train_losses)

                if cur_count == batchs:
                    print(cur_epoch)

                    losses_per_batch = np.array(losses_per_batch)
                    # print(losses_per_batch)
                    losses_per_batch = losses_per_batch.mean(axis=0)
                    print(losses_per_batch)

                    losses_per_epoch.append(losses_per_batch)

                    cur_count = 0
                    losses_per_batch = []

            else:
                for i in range(3):
                    f.readline()
                val_scores = f.readline().strip().split("|")
                # ['', ' t2i  ', ' 57.100 ', ' 79.350 ', ' 87.600 ', ' 46.298 ', ' 25.333 ', '']

                val_scores = [float(i) for i in val_scores[2:7]]
                # [57.100, 79.350, 87.600, 46.298, 25.333]
                print(val_scores)
                val_scores_per_epoch.append(val_scores)
print(losses_per_epoch)
print(val_scores_per_epoch)

for i,j in zip(losses_per_epoch,val_scores_per_epoch):
    print(i,j,sep="\n")
    print()

