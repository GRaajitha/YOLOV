import os
from yolox.exp import get_exp
import argparse
import torch
import cv2
from yolox.data.data_augment import preproc_no_pad
import onnx
import numpy as np
import onnxruntime as ort

def make_parser():
    parser = argparse.ArgumentParser(description="YOLOV ONNX Export Parser")
    parser.add_argument("-f", "--exp_file", default='', type=str, help="Path to experiment description file")
    parser.add_argument("-d", "--outdir", default='./onnx_export', type=str, help="Output directory")
    parser.add_argument("-c", "--ckpt", default='', type=str, help="Path to checkpoint file")
    parser.add_argument("-o", "--out_file", default='', type=str, help="Path for output ONNX file")
    parser.add_argument("-n", "--name", type=str, default=None, help="Model name")
    return parser

def load_model(args):
    """
    load model and state dict from arguments
    """
    exp = get_exp(args.exp_file, args.name)
    exp.onnx_export = True

    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()
    return model, exp

def visualize_detections(pre_proc_img, detections, outdir):
    """Visualizes inference results and saves the output images."""
    # Convert image format
    img = pre_proc_img.astype('uint8').transpose(1, 2, 0)
    img = np.ascontiguousarray(img)
    cv2.imwrite(f"{outdir}/preprocessed_input.png", img)
    
    for detection in detections:
        # xmin, ymin, xmax, ymax, obj_score, cls_score, cls = detection[:7]
        xmin, ymin, xmax, ymax, obj_score = detection[:5]
        class_scores = detection[5:]
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        score = obj_score * max(class_scores)
        cls = np.argmax(class_scores).item()

        if score >= 0.5:
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            img = cv2.putText(
                img, f"cls:{cls}__score:{score:.3f}", (xmin - 10, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2
            )
    
    output_filename = "onnx_inference.png"
    cv2.imwrite(f"{outdir}/{output_filename}", img)
    print(f"Visualization of inference saved as {output_filename}!")


if __name__ == "__main__":
    args = make_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    output_onnx_file = f"{args.outdir}/{args.out_file}"
    
    model, exp = load_model(args)

    output_names = ["output"]
    input_names = ["input"]

    # prepare input
    img_cv = cv2.imread("./input_image.png")
    pre_proc_img, _, _ = preproc_no_pad(img_cv, (exp.input_size[0], exp.input_size[1]))
    model_input = torch.tensor(pre_proc_img)
    model_input = model_input.unsqueeze(0).cuda()
    inputs = {"x": model_input}

    data = np.asarray(model_input.cpu().numpy(), dtype=np.float32)
    data.tofile(f"{args.outdir}/preproc_single_1920_input.dat")
    
    #export model to onnx
    torch.onnx.export(
        model,  # model being run
        inputs,  # model input (or a tuple for multiple inputs)
        output_onnx_file,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=13,  # ONNX version - v13 supports channel level quantization.
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=input_names,  # the model's input names
        output_names=output_names,  # the model's output names
        # Not using dynamic axes for now
        # dynamic_axes={
        #     "input": {0: "batch_size"},
        #     "output": {0: "batch_size"},
        # },
        verbose=True,
    )
    print(f"ONNX file written to {output_onnx_file}")
    # verify onnx export with onnx checker
    onnx_model = onnx.load(output_onnx_file)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(output_onnx_file)
    input_names = [inp.name for inp in session.get_inputs()] 
    output_names = [out.name for out in session.get_outputs()]
    ort_inputs = {input_names[i]: v.cpu().numpy() for i, v in enumerate(inputs.values())}
    outputs = session.run(output_names, ort_inputs)

    visualize_detections(pre_proc_img, outputs[0], args.outdir)