from vit_pytorch import ViT
from util import get_num_parameters

if __name__ == "__main__":
    print(get_num_parameters(ViT(image_size = 32, patch_size = 16, num_classes = 100,
                                 dim = 512, depth = 6, heads = 8, mlp_dim = 1024, dim_head = 64)))