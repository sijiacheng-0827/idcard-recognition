import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.transforms.functional as functional
from torchvision import models
import time

# 预处理
image = Image.open(r'C:\Users\Administrator\Desktop\新建文件夹\front_cp.bmp')
image = functional.crop(image, left=0, top=0, width=648, height=648)
transform = transforms.Compose([transforms.Resize(224),transforms.RandomRotation(10),transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) ])
transed_image = transform(image)
transed_image = transed_image.unsqueeze(0)

# 初始化模型、损失函数和优化器

model = models.resnet18()
num_classes = 3
model.fc = nn.Linear(model.fc.in_features,num_classes)
model.load_state_dict(torch.load(r'F:\idcard\result\idcard_model_19.pth'))
model.eval()



# # 保存模型
# dummy_input = torch.rand(1,3,224,224)
# with torch.no_grad():
#  jit_model = torch.jit.trace(model,dummy_input)
#  jit_model.save("F:/model.pt")
# load_net = torch.jit.load("F:/model.pt","cpu")
# print(load_net)



# 训练模型
start_time = time.time()
output = model(transed_image)
_, predicted = torch.max(output.data, 1)
if predicted == 0:
    print('正面')
elif predicted == 1:
    print('反面')
else:
    print('遮挡过度')
end_time = time.time()
total_time = end_time - start_time
print("程序运行时间：", total_time, "秒")


