# ADVERSARIAL EXAMPLES IN THE PHYSICAL WORLD 工具复现说明文档

## 文件说明

+ 实现了Adversarial.py文件，实现对论文中关键过程的复现

+ 在Adversarial.py文件中，path为所要识别的文件夹路径，phphotos为预先将对抗样例使用手机摄像头拍摄并将其传到pc端的图片输入文件夹，其中的文件名须与path中对应图片的文件名保持一致

### class.txt文件

+ 该文件保存了InceptionV3模型对应的1000个类别信息，训练的默认类别为波斯猫，如果需要训练其他类别，需要根据class.txt文件中类别对应id更改语句‘perturbations = get_perturbations(283, img_probs, img)’中的参数283为相应id

### py文件实现了以下功能

+ 对于给定的path（对应图片文件夹），使用inception模型预测其中所有图片的类别

+ 对于每一张图片，使用论文中提到的FGSM、BIM、ll方法对其进行对抗样本训练（对应fgsm_train/bim_train/least_likely_class方法)

+ 在上述每一个方法中，会分别打印扰动对应的像素图片、添加扰动后的对抗样本图片结果以及对应的参数（ε、α）

+ 在每种训练方法中，将用top-5可能的类是否与原图片类别符合来评估训练的效果，结果将在控制台打印

+ 训练的默认类别为波斯猫（Persian_cat)
+ 具体ａｐｉ涵义已经在文件中标出

