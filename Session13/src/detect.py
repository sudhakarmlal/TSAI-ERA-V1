from typing import List
import cv2
import torch
import numpy as np
import src.config as config
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.model_obj import Assignment13
from src.utils import cells_to_bboxes, non_max_suppression, draw_predictions, YoloCAM




weights_path = "/home/user/app/model_ass_13_up.ckpt"
model = Assignment13().load_from_checkpoint(weights_path,map_location=torch.device("cpu"))
model = model.model
#ckpt = torch.load(weights_path, map_location="cpu")
#model.load_state_dict(ckpt)
model.eval()
print("[x] Model Loaded..")

scaled_anchors = (
    torch.tensor(config.ANCHORS)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to(config.DEVICE)

cam = YoloCAM(model=model, target_layers=[model.layers[-2]], use_cuda=False)

def predict(image: np.ndarray, iou_thresh: float = 0.5, thresh: float = 0.4, show_cam: bool = False, transparency: float = 0.5) -> List[np.ndarray]:
    with torch.no_grad():
        transformed_image = config.transforms(image=image)["image"].unsqueeze(0)
        output = model(transformed_image)
        
        bboxes = [[] for _ in range(1)]
        for i in range(3):
            batch_size, A, S, _, _ = output[i].shape
            anchor = scaled_anchors[i]
            boxes_scale_i = cells_to_bboxes(
                output[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

    nms_boxes = non_max_suppression(
        bboxes[0], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
    )
    plot_img = draw_predictions(image, nms_boxes, class_labels=config.PASCAL_CLASSES)
    if not show_cam:
        return [plot_img]
    
    grayscale_cam = cam(transformed_image, scaled_anchors)[0, :, :]
    img = cv2.resize(image, (416, 416))
    img = np.float32(img) / 255
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True, image_weight=transparency)
    return [plot_img, cam_image]


