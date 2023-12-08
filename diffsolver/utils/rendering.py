from .misc import MultiToolEnv, WorldState
import numpy as np
from numpy.typing import NDArray
from typing import Mapping, Union, List, Any
from PIL import Image, ImageDraw, ImageFont

def text_to_image(text, image_size=(512, 512), bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    # Create a new image with the specified background color
    img = Image.new('RGB', image_size, bg_color)

    # Create an ImageDraw object
    draw = ImageDraw.Draw(img)

    # Optionally, specify a font size and type. If you don't have the 'arial.ttf' font file, 
    # you can use a different one, or just comment out this line to use the default font.
    font = ImageFont.truetype('DejaVuSans.ttf', 15)

    # Split the text into lines
    lines = text.split('\n')

    y_text = 10
    for line in lines:
        # Draw the text onto the image
        draw.text((10, y_text), line, font=font, fill=text_color)
        y_text += 15

    return img


def create_image_with_titles(images, titles, dir='below', width=200):
    images = [Image.fromarray(image) for image in images]
    # Calculate the dimensions of the output image
    num_images = len(images)
    image_width, image_height = images[0].size
    if dir == 'below':
        output_width = image_width * num_images
        output_height = image_height + 60
    elif dir == 'left':
        output_width = image_width + width
        output_height = image_height * num_images
    elif dir == 'above':
        output_width = image_width * num_images
        output_height = image_height + 60
    else:
        raise NotImplementedError

    # Create a new image to hold all of the input images and titles
    output_image = Image.new('RGB', (output_width, output_height), color='white')

    # Add each input image and title to the output image
    font = ImageFont.truetype('DejaVuSans.ttf', size=40)
    draw = ImageDraw.Draw(output_image)
    for i in range(num_images):
        image = images[i]
        _font = font
        if 'FONT:' in titles[i]:
            title, font_size = titles[i].split('FONT:')
            title = title.strip()
            _font = ImageFont.truetype('DejaVuSans.ttf', size=int(font_size))
        else:
            title = str(titles[i])
        title_size = draw.textsize(title, font=_font)
        if dir == 'below':
            x = i * image_width
            y = 0
            output_image.paste(image, (x, y))

            x += (image_width - title_size[0]) // 2
            y += image_height + 5
        elif dir == 'above':
            x = i * image_width
            y = 5 + 50
            output_image.paste(image, (x, y))

            x += (image_width - title_size[0]) // 2
            y = 0
        elif dir == 'left':
            x = i * image_width + width
            y = 0
            output_image.paste(image, (x, y))

            x = 0 + 5
            y = 0 +  (image_height - title_size[1]) // 2
        else:
            raise NotImplementedError

        draw.text((x, y), title, font=_font, fill=(0, 0, 0))

    return np.array(output_image)


def rendering_objects(env: MultiToolEnv, objects: Mapping[str, np.ndarray], scene_title='scene'):
    state: WorldState = env.get_state()
    img: Any = env.render('rgb_array')
    assert img is not None
    images = [img]
    for v in objects.values():
        pos = state.X[v]
        assert state.color is not None
        color = state.color[v]
        img = env.render_state_rgb(pos = pos, color=color)
        images.append(img)

    return create_image_with_titles(images, [scene_title] + list(objects.keys()))