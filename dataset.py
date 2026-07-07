import random
import io
from PIL import Image

class BalancedJPEGCompression(object):
    def __init__(self, trigger_prob=0.5, quality_range=(40, 85)):
        self.trigger_prob = trigger_prob
        self.quality_range = quality_range

    def __call__(self, img):
        if random.random() < self.trigger_prob:
            quality = random.randint(*self.quality_range)
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=quality)
            output.seek(0)
            img = Image.open(output)
        return img