from .mnist_one_vs_rest import SimpleIsThree
from .celeba import CelebA_binary
from .bg_bird import BgBird, RedBgBlueBird
from .shapenet import ShapeNet


def get_dataset(data_root, dataset_name):
    if dataset_name == "simpleisthree":
        return SimpleIsThree(data_root)
    elif dataset_name.startswith("celeba_"):
        target_attribute = "_".join(dataset_name.split("_")[1:])
        return CelebA_binary(data_root, target_attribute)
    elif dataset_name == "bg_bird":
        return BgBird(data_root)
    elif dataset_name == "redbg_bluebird":
        return RedBgBlueBird(data_root)
    elif dataset_name == "shapenet":
        return ShapeNet(data_root)
    else:
        raise NotImplementedError