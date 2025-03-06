import torch
import torchvision

class CustomNMSFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, boxes, scores, labels, iou_threshold, score_threshold, max_predictions_per_image):
        # Perform batched NMS (handles multi-class NMS)
        indices = torchvision.ops.batched_nms(boxes[0], scores[0][0], labels, iou_threshold)
        a = torch.randn((indices.shape[0], 1))
        b = torch.randn((indices.shape[0], 1))
        c = torch.randn((indices.shape[0], 1))
        d = torch.randn((indices.shape[0], 1))
        return torch.cat([a, b, c, d], dim=1)

    @staticmethod
    def symbolic(g, boxes, scores, labels, iou_threshold, score_threshold, max_predictions_per_image):
        # This is a symbolic definition that maps to an ONNX op

        print(boxes, scores, labels, iou_threshold, score_threshold, max_predictions_per_image)
        return g.op(
            "NonMaxSuppression", 
            boxes, 
            scores,
            max_predictions_per_image,
            iou_threshold,
            score_threshold,
        )