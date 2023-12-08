# from PIL import Image
# import os

# # 指定图片目录
# img_dir = 'figues'

# # 读取图片目录下所有文件
# imgs = [Image.open(os.path.join(img_dir, img)) for img in os.listdir(img_dir) if img.endswith(".png")]

# for img in os.listdir(img_dir):
#     if img.endswith(".png"):
#         img = img.replace('.png','')
#         print(img)

# # 确保图片数量
# # assert len(imgs) == 81, "Need exactly 81 images!"

# # 获取单个图片的宽度和高度
# img_width, img_height = imgs[0].size

# # 创建一个新的空白图片，大小为9*单张图片的宽度，9*单张图片的高度
# total_width = img_width * 4
# total_height = img_height * 4
# new_img = Image.new('RGB', (total_width, total_height))

# # 按照7x7的网格布局粘贴图片
# for i in range(4):
#     for j in range(4):
#         img = imgs[i*4 + j]
#         new_img.paste(img, (j*img_width,i*img_height))

# # 保存新图片
# new_img.save('combined.png')
import os
from PIL import Image

# 获取目录下所有的png图片
folder_path = "figues"
images = [img for img in os.listdir(folder_path) if img.endswith(".png")]

# 确定每个区域图像的数量
area_images_count = [9, 16, 25, 36]

# 计算每个区域图像的边长（即每个区域应放置的图像数量的平方根）
area_side_lengths = [int(i ** 0.5) for i in area_images_count]

# 计算需要的大图像的大小
single_image_size = Image.open(os.path.join(folder_path, images[0])).size
single_width, single_height = single_image_size

# 创建一个空白的大图像
new_image = Image.new('RGB', (single_width*6, single_height*6))

# 将小图像按照要求放入大图像中
index = 0
for area_index in range(4):
    width_start = 0 if area_index < 2 else single_width*3
    height_start = 0 if area_index%2 == 0 else single_height*3
    for i in range(area_side_lengths[area_index]):
        for j in range(area_side_lengths[area_index]):
            img = Image.open(os.path.join(folder_path, images[index]))
            img = img.resize((single_width*3 // area_side_lengths[area_index], 
                                single_height*3 // area_side_lengths[area_index]))
            new_image.paste(img, ((width_start+j*single_width*3 // area_side_lengths[area_index]),
                                 (height_start+i*single_height*3 // area_side_lengths[area_index])))
            index += 1

# 保存大图像

new_image = new_image.resize((single_width*2, single_height*2))
new_image.save("image.png")