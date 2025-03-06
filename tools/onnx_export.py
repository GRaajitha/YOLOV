import os
from yolox.exp import get_exp
import argparse
import torch
import cv2
from yolox.data.data_augment import preproc

def make_parser():
    parser = argparse.ArgumentParser("YOLOV onnx export parser")
    parser.add_argument(
        "-f",
        "--exp_file",
        default='',
        type=str,
        help="input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default='', type=str, help="checkpoint file")
    parser.add_argument(
        "-o",
        "--out_file",
        default='',
        type=str,
        help="path for output onnx file",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    return parser

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.onnx_export = True
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # resume the model/optimizer state dict
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda")
    model.cuda()
    img_cv = cv2.imread("/home/rgummadi/YOLOV/input_image.png")
    input, _ = preproc(img_cv, (exp.input_size[0], exp.input_size[1]))
    input = torch.tensor(input)
    input = torch.stack([input] * exp.gframe, dim=0).cuda()
    # rand_img = torch.rand((exp.gframe, 3, exp.input_size[1], exp.input_size[0])).cuda()

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,  # model being run
            input,  # model input (or a tuple for multiple inputs)
            args.out_file,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=13,  # ONNX version - v13 supports channel level quantization.
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input"],  # the model's input names
            output_names=["output"],  # the model's output names
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size", 1: str(exp.defualt_pre)},
            },
            verbose=True,
        )
        print(f"Written ONNX file to {args.out_file}")

