import os
from yolox.exp import get_exp
import argparse
import torch
import cv2
from yolox.data.data_augment import preproc_no_pad
import onnx
import numpy as np

def make_parser():
    """Creates an argument parser for YOLOV ONNX export."""
    parser = argparse.ArgumentParser(description="YOLOV ONNX Export Parser")
    
    parser.add_argument("-f", "--exp_file", default='', type=str, help="Path to experiment description file")
    parser.add_argument("-d", "--outdir", default='./onnx_export', type=str, help="Output directory")
    parser.add_argument("-c", "--ckpt", default='', type=str, help="Path to checkpoint file")
    parser.add_argument("-o", "--out_file", default='', type=str, help="Path for output ONNX file")
    parser.add_argument("--backbone_only", action="store_true", default=False, help="Export only YOLOV++ backbone")
    parser.add_argument("--head_only", action="store_true", default=False, help="Export only YOLOV++ head")
    parser.add_argument("-n", "--name", type=str, default=None, help="Model name")
    parser.add_argument("-fd", "--fpn_dir", type=str, default=None, help="Directory containing fpn outputs from backbone only model")
    
    return parser

def visualize_and_save_fpn_outputs(outputs, save_dir="fpn_visualizations"):
    """
    Saves FPN feature maps as images.

    Args:
        outputs (list of numpy arrays): FPN outputs from YOLOX.
        save_dir (str): Directory to save the images.
    """
    os.makedirs(save_dir, exist_ok=True)  # Create directory if not exists
    fpn_levels = ['fpn_out0', 'fpn_out1', 'fpn_out2']
    for i, output in enumerate(outputs):
        output[0].astype(np.float32).tofile(f"{save_dir}/{fpn_levels[i]}.dat")
        feature_map = output[0, 0, :, :]  # Extract first channel
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-6)
        feature_map = (feature_map * 255).astype(np.uint8)
        # Resize to the highest resolution (136x240) for consistency
        resized_map = cv2.resize(feature_map, (240, 136), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{save_dir}/{fpn_levels[i]}.png", resized_map)
    print("saved fpn outs")

def visualize_detections(pre_proc_img, outputs, input_shape, outdir):
    """Visualizes inference results and saves the output images."""
    # Convert image format
    img = pre_proc_img.astype('uint8').transpose(1, 2, 0)
    img = np.ascontiguousarray(img)
    cv2.imwrite(f"{outdir}/preprocessed_input.png", img)

    # Determine output structure
    detections = outputs[0][0] if input_shape[0] != 1 else outputs[0]
    
    for detection in detections:
        xmin, ymin, xmax, ymax, obj_score, cls_score, cls = detection[:7]
        xmin, ymin, xmax, ymax, cls = map(int, [xmin, ymin, xmax, ymax, cls])
        score = obj_score * cls_score
        
        if score >= 0.5:
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            img = cv2.putText(
                img, f"cls:{cls}__score:{score:.3f}", (xmin - 10, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2
            )
    
    output_filename = (
        "onnx_inference_0.5score_thresh.png" if input_shape[0] != 1 
        else "inference_single_input_pre_nms_0.5score_thresh.png"
    )
    cv2.imwrite(f"{outdir}/{output_filename}", img)
    print(f"Visualization of inference saved as {output_filename}!")

def load_model(args):
    """
    load model and state dict from arguments
    """
    exp = get_exp(args.exp_file, args.name)
    exp.onnx_export = True
    exp.defualt_pre = 100
    exp.backbone_only = args.backbone_only
    exp.head_only = args.head_only
    assert not(exp.backbone_only == True and exp.head_only==True), "both backbone_only and head_only options are mutually exclusive"

    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()
    return model, exp

def load_fpn_outputs(exp, base_path, out_dir):
    """Loads precomputed FPN outputs from files."""
    fpn_shapes = {"fpn_out0": exp.fpn0_shape, "fpn_out1": exp.fpn1_shape, "fpn_out2": exp.fpn2_shape}
    fpn_outputs = {}

    for i, shape in enumerate(fpn_shapes):
        fpn_name = f"fpn_out{i}"
        file_path = os.path.join(base_path, f"{fpn_name}.dat")
        fpn_out = torch.tensor(np.fromfile(file_path, dtype=np.float32))
        fpn_out = fpn_out.reshape(fpn_shapes[fpn_name])
        fpn_out = torch.stack([fpn_out] * exp.gframe, dim=0).cuda()
        fpn_out_data = np.asarray(fpn_out.cpu().numpy(), dtype=np.float32)
        fpn_out_data.tofile(f"{out_dir}/batch{exp.gframe}_{fpn_name}.dat")
        fpn_outputs[fpn_name] = fpn_out

    return fpn_outputs

def make_inputs(pre_proc_img):
    input = torch.tensor(pre_proc_img)
    if exp.backbone_only:
        input = input.unsqueeze(0).cuda()
        data = np.asarray(input.cpu().numpy(), dtype=np.float32)
        data.tofile(f"{args.outdir}/preproc_single_1920_input.dat")
    else:
        input = torch.stack([input] * exp.gframe, dim=0).cuda()
        data = np.asarray(input.cpu().numpy(), dtype=np.float32)
        data.tofile(f"{args.outdir}/preproc_batch{exp.gframe}_1920_input.dat")
    # rand_img = torch.rand((exp.gframe, 3, exp.input_size[1], exp.input_size[0])).cuda()
    return input

def export_to_onnx(model, output_onnx_file, fpn_dir, model_input):
    output_names = ["output"]
    input_names = ["input"]
    inputs = {"x": model_input}
    if exp.backbone_only:
        output_names = ["fpn_out0", "fpn_out1", "fpn_out2"]
    if exp.head_only:
        input_names = ["fpn_out0", "fpn_out1", "fpn_out2"]
        inputs = load_fpn_outputs(exp, base_path=fpn_dir, out_dir=args.outdir)
    
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
    return inputs

if __name__ == "__main__":
    args = make_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    output_onnx_file = f"{args.outdir}/{args.out_file}"

    #load model and state dict from checkpoint
    model, exp = load_model(args)

    # prepare input
    img_cv = cv2.imread("./input_image.png")
    pre_proc_img, _ = preproc_no_pad(img_cv, (exp.input_size[0], exp.input_size[1]))
    model_input = make_inputs(pre_proc_img)

    # Export model to onnx
    ort_inputs = export_to_onnx(model, output_onnx_file, args.fpn_dir, model_input)

    # load model with onnxruntime and verify outputs
    import onnxruntime as ort
    session = ort.InferenceSession(output_onnx_file)
    input_names = [inp.name for inp in session.get_inputs()] 
    output_names = [out.name for out in session.get_outputs()]
    ort_inputs = {input_names[i]: v.cpu().numpy() for i, v in enumerate(ort_inputs.values())}
    outputs = session.run(output_names, ort_inputs)

    #Visualization
    if exp.backbone_only:
        visualize_and_save_fpn_outputs(outputs, args.outdir)
    else:
        visualize_detections(pre_proc_img, outputs, input.shape, args.outdir)