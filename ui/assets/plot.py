from PIL import Image
import os


# 读取图片目录下所有文件
img = Image.open(os.path.join('Ground_Texture.jpg'))

print(img)
img_width, img_height = (256, 256)

total_width = img_width * 20
total_height = img_height * 20
new_img = Image.new('RGB', (total_width, total_height))

# 按照7x7的网格布局粘贴图片
for i in range(20):
    for j in range(20):
        new_img.paste(img, (j*img_width,i*img_height))

# 保存新图片
new_img.save('combined.png')