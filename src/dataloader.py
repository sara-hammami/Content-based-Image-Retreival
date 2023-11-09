from PIL import Image
import pathlib

from torchvision import transforms
from torch.utils.data import Dataset
allowed_extensions = ['.jpg', '.png']
def get_transformation():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation = transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    return transform

class MyDataLoader(Dataset):

    def __init__(self, image_root):
        self.image_root = pathlib.Path(image_root)
        self.image_list = list()
        for image_path in self.image_root.iterdir():
            if image_path.exists() and image_path.suffix.lower() in allowed_extensions:
                self.image_list.append(image_path)
        self.image_list = sorted(self.image_list, key = lambda x: x.name)
        self.transform = get_transformation()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        _img = self.image_list[index]
        _img = Image.open(_img)
        _img = _img.convert("RGB")
        return self.transform(_img), str(self.image_list[index])