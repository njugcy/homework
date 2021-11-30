import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
# 初始化模型
mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False
pretrained_model = tf.keras.applications.InceptionV3(include_top=True,
                                                     weights='imagenet')
pretrained_model.trainable = False
# 照片地址
path = "photos"
# 将准备好的映射表写入成dict
f = open("class.txt")
classes = f.readlines()
class_dict = {}
for i in range(1000):
    cl = classes[i].split(":")
    cl[1] = cl[1][2:len(cl[1])-3]
    cl[1] = cl[1].split(',')[0]
    cl[1] = cl[1].replace(" ", "_")
    class_dict[cl[1]] = int(cl[0])


# 图像预处理，使其符合inceptionv3的默认输入尺寸
def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = image/255
    image = tf.image.resize(image, (299, 299))
    image = image[None, ...]
    return image


# 将path读入为合法的image图像
def parsepath(path):
    image_raw = tf.io.read_file(path,'rb')
    image = tf.image.decode_image(image_raw)
    image = preprocess(image)
    return image


# 提取最高可能的预测标签
def get_imagenet_label(probs):
    return tf.keras.applications.inception_v3.decode_predictions(probs, top=1)[0][0]


# 提取前五可能的预测标签集合
def get_top_five(probs):
    tp5 = tf.keras.applications.inception_v3.decode_predictions(probs, top=5)
    return tp5[0]


# 提取最不可能的预测标签（yLL）
def get_least(probs):
    tp1000 = tf.keras.applications.inception_v3.decode_predictions(probs, top=1000)
    return tp1000[0][999]


# 根据class_dict获取标签对应id
def get_class_index(label):
    return class_dict[label]


# 输出原图的预测值
def print_or(img):
    image_probs = pretrained_model.predict(img)
    plt.figure()
    plt.imshow(img[0])
    _, image_class, class_confidence = get_imagenet_label(image_probs)
    plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
    plt.show()


loss_object = tf.keras.losses.CategoricalCrossentropy()


# 根据损失函数计算梯度，根据梯度下降方法返回上升的梯度
def create_adversarial_pattern(input_image, input_label):

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


# 计算扰动
def get_perturbations(index, image_probs, img):
    label = tf.one_hot(index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    perturbations = create_adversarial_pattern(img, label)

    return perturbations


# 输出图像
def display_images(image, description):
    _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
    plt.figure()
    plt.imshow(image[0])
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
    plt.show()


# FGSM方法
def fgsm_train(img):
    epsilon = 0.02
    description = 'Epsilon = {:0.3f}'.format(epsilon)
    img_probs = pretrained_model.predict(img)
    perturbations = get_perturbations(283,img_probs, img)
    plt.imshow(perturbations[0])
    adv_x = img + epsilon*perturbations  # Persian cat
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    display_images(adv_x, description)
    return calculate_accuracy(adv_x)


# BIM方法
def bim_train(img):
    alpha = 1/255
    img_probs = pretrained_model.predict(img)
    perturbations = get_perturbations(283, img_probs,img)
    plt.imshow(perturbations[0])
    adv_x = img + alpha*perturbations
    for i in range(6):  # min{1.25ε, ε+4}
        adv_x = adv_x + alpha*perturbations
        adv_x = tf.clip_by_value(adv_x, 0, 1)
    display_images(adv_x, "BMI: Alpha = 1")
    return calculate_accuracy(adv_x)


# l.l.方法
def least_likely_class(img):
    alpha = 1/255
    img_probs = pretrained_model.predict(img)
    least = get_least(img_probs)
    label = least[1]
    index = get_class_index(label)
    perturbations = get_perturbations(index, img_probs, img)
    plt.imshow(perturbations[0])
    adv_x = img - alpha * perturbations
    for i in range(6):
        adv_x = adv_x - alpha*perturbations
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    display_images(adv_x, "l.l.: Alpha = 1")
    return calculate_accuracy(adv_x)


# 计算准确率
def calculate_accuracy(img):
    top5 = get_top_five(pretrained_model.predict(img))
    accuracy = False
    for t in top5:
        if t[1] == "Persian_cat":
            accuracy = True
    return accuracy


# 遍历photos文件夹下的文件执行预测并统计结果输出
count = 0
original = 0
fgsm = 0
bim = 0
ll = 0
for root, dirs, files in os.walk(path):
    for f in files:
        count += 1
        img1 = parsepath(root + '\\' + f)
        img2 = parsepath('phphotos'+'\\'+f)
        print_or(img1)
        if calculate_accuracy(img1):
            original += 1
        if fgsm_train(img1):
            fgsm += 1
        if bim_train(img1):
            bim += 1
        if least_likely_class(img1):
            ll += 1
        display_images(img2, "adversarial version:")
print('Original = {:0.3f}'.format(original/count))
print('FGSM = {:0.3f}'.format(fgsm/count))
print('BIM = {:0.3f}'.format(bim/count))
print('l.l. = {:0.3f}'.format(ll/count))
