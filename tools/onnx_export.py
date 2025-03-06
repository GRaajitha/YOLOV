import os
from yolox.exp import get_exp
import argparse
import torch
import cv2
from yolox.data.data_augment import preproc
import onnx
import numpy as np

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

    #load model and state dict from checkpoint
    exp = get_exp(args.exp_file, args.name)
    exp.onnx_export = True
    exp.defualt_pre = 100
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda")
    model.cuda()

    # prepare input
    img_cv = cv2.imread("/home/rgummadi/YOLOV/input_image.png")
    pre_proc_img, _ = preproc(img_cv, (exp.input_size[0], exp.input_size[1]))
    input = torch.tensor(pre_proc_img)
    # input = input.unsqueeze(0).cuda()
    # data = np.asarray(input.cpu().numpy(), dtype=np.float32)
    # data.tofile("preproc_single_1920_input.dat")
    input = torch.stack([input] * exp.gframe, dim=0).cuda()
    data = np.asarray(input.cpu().numpy(), dtype=np.float32)
    data.tofile("preproc_batch4_1920_input.dat")
    rand_img = torch.rand((exp.gframe, 3, exp.input_size[1], exp.input_size[0])).cuda()
            
    #export model to onnx
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
            # dynamic_axes={
            #     "input": {0: "batch_size"},
            #     "output": {0: "batch_size"},
            # },
            verbose=True,
        )
        print(f"ONNX file written to {args.out_file}")

    # verify onnx export with onnx checker
    onnx_model = onnx.load(args.out_file)
    onnx.checker.check_model(onnx_model)

    # load model with onnxruntime and verify outputs
    import onnxruntime as ort
    session = ort.InferenceSession(args.out_file)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input.cpu().numpy()})

    #visualize a inferences on single image
    img = pre_proc_img.astype('uint8')  # Convert to uint8
    # Convert from (C, H, W) to (H, W, C)
    img = img.transpose(1, 2, 0)
    img = np.ascontiguousarray(img)
    
    if input.shape[0] != 1:
        for detection in outputs[0][0]:
            xmin, ymin, xmax, ymax, obj_score, cls_score, cls = detection
            xmin, ymin, xmax, ymax, cls = map(int, [xmin, ymin, xmax, ymax, cls])
            score = obj_score * cls_score
            if score >= 0.5:
                img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 3)
                img = cv2.putText(img, f"cls:{cls}__score:{score:.3f}", (xmin-10, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
        
        cv2.imwrite(os.path.join("./onnx_inference_0.5score_thresh.png"), img)
        print("Visualization of inference saved!")
    else:
        for detection in outputs[0]:
            xmin, ymin, xmax, ymax, obj_score, cls_score, cls = detection[:7]
            xmin, ymin, xmax, ymax, cls = map(int, [xmin, ymin, xmax, ymax, cls])
            score = obj_score * cls_score
            if score >= 0.5:
                img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 3)
                img = cv2.putText(img, f"cls:{cls}__score:{score:.3f}", (xmin-10, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
        cv2.imwrite(os.path.join("./inference_single_input_pre_nms_0.5score_thresh.png"), img)
        print("Visualization of inference saved!")
