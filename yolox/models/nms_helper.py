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


# import torch
# import torchvision


# class CustomNMSFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
#         idxs = torch.randn(boxes.shape[1])
#         indices = torchvision.ops.batched_nms(boxes[0], scores[0][0], idxs, iou_threshold)
#         a = torch.randn((indices.shape[0], 1))
#         b = torch.randn((indices.shape[0], 1))
#         c = torch.randn((indices.shape[0], 1))
#         d = torch.randn((indices.shape[0], 1))
#         return torch.cat([a, b, c, d], dim=1)

#     @staticmethod
#     def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
#         # In the symbolic method, you should define the ONNX-compatible
#         # version of your custom operator.
#         output = g.op(
#             "NonMaxSuppression",
#             boxes,
#             scores,
#             max_output_boxes_per_class,
#             iou_threshold,
#             score_threshold,
#         )
#         return output


    # @staticmethod
    # def symbolic(g, boxes, scores, labels, max_output_boxes_per_class, iou_threshold, score_threshold):
    #     """
    #     Symbolic function for ONNX export.
    #     Ensures per-class NonMaxSuppression by expanding scores.
    #     """
    #     g.op("Print", boxes.shape)
    #     g.op("Print", scores.shape) 
    #     g.op("Print", labels.shape) 
    #     g.op("Print", max_output_boxes_per_class)
    #     g.op("Print", iou_threshold)
    #     g.op("Print", score_threshold)
    #     # Unsqueeze boxes to match ONNX expected shape: (1, N, 4)
    #     boxes = g.op("Unsqueeze", boxes, g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)))

    #     # Process scores per class
    #     num_classes = 1 + g.op("Max", labels)  # Compute max class ID to determine number of classes
    #     scores = g.op("Unsqueeze", scores, g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)))  # Shape (1, N)
    #     scores = g.op("Unsqueeze", scores, g.op("Constant", value_t=torch.tensor([2], dtype=torch.int64)))  # Shape (1, 1, N)
        
    #     # Convert labels to one-hot encoding (shape: (N, num_classes))
    #     labels_one_hot = g.op("OneHot", labels, num_classes, g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)))

    #     # Multiply scores with one-hot labels to split across class dimension (1, num_classes, N)
    #     class_scores = g.op("Mul", scores, g.op("Unsqueeze", labels_one_hot, g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))))

    #     # Ensure max_output_boxes_per_class, iou_threshold, and score_threshold are tensors
    #     max_output_boxes_per_class = g.op("Unsqueeze", max_output_boxes_per_class, g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)))
    #     iou_threshold = g.op("Unsqueeze", iou_threshold, g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)))
    #     score_threshold = g.op("Unsqueeze", score_threshold, g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)))

    #     # Run ONNX NonMaxSuppression
    #     selected_indices = g.op(
    #         "NonMaxSuppression",
    #         boxes,
    #         class_scores,
    #         max_output_boxes_per_class,
    #         iou_threshold,
    #         score_threshold,
    #         outputs=1  # Return selected indices
    #     )

    #     return selected_indices