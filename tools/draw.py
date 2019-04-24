from tools.basic import xywh2ullr
from PIL import Image, ImageDraw


def draw_box(img_pil, ul=None, lr=None, xywh=None, xy=None, wh=None, fill=(0,0,0), width=2):
    """Draw a box in a PIL image from 
    * upper-left (ul) to lower-right (lr) or 
    * using xywh=(x-center, y-center, width, height), or
    * using xy=(x-center, y-center) and wh=(width, height).
    
    TODO warning when outside image.
    TODO parameter for: reshape box when outside image."""
    if ul or lr:
        assert all(v is None for v in [xywh, xy, wh]), "When you use 'ul' and 'lr' all the other postion parameters must be 'None'."
    if xywh:
        assert all(v is None for v in [ul, lr, xy, wh]), "When you use 'xywh' all the other postion parameters must be 'None'."
    if xy or wh:
        assert all(v is None for v in [ul, lr, xywh]), "When you use 'xy' and 'wh' all the other postion parameters must be 'None'."
        xywh = xy + wh
    if xywh or xy:
        #ul = (xywh[0]-.5*xywh[2], xywh[1]-.5*xywh[3])
        #lr = (xywh[0]+.5*xywh[2], xywh[1]+.5*xywh[3])
        ul, lr = xywh2ullr(xywh)
    
    diff_width = lr[0] - ul[0]
    diff_height = lr[1] - ul[1]

    ur = (ul[0] + diff_width, ul[1])
    ll = (ul[0], ul[1] + diff_height)
    draw = ImageDraw.Draw(img_pil)
    draw.line(ul + ur, fill=fill, width=width)
    draw.line(ur + lr, fill=fill, width=width)
    draw.line(lr + ll, fill=fill, width=width)
    draw.line(ll + ul, fill=fill, width=width)
    return img_pil

def draw_points(img, xys, size = 3, color_inner = 'yellow', color_outer='black'):
    """Draw point(s) on a PIL image.

    TODO warning when outside image."""
    if isinstance(xys, tuple):
        xys = [xys]
    for xy in xys:
        draw = ImageDraw.Draw(img)
        draw.ellipse((xy[0]-size, xy[1]-size, xy[0]+size, xy[1]+size), fill=color_inner, outline=color_outer)
    return img