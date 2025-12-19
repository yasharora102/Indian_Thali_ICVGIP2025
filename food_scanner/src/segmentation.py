import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


    
    
class Segmenter:
    def __init__(self, cfg: dict, device: torch.device):
        
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            print("Using Ampere GPU, enabling tf32")
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            
        sam = build_sam2(cfg['sam_cfg'], cfg['sam_ckpt'], device)
        self.sam_pred = SAM2ImagePredictor(sam)
        model_id = "IDEA-Research/grounding-dino-base"
        # model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.device = device
        self.num_params_gdino = sum(p.numel() for p in self.gdino.parameters() if p.requires_grad)
        self.num_params_sam = sum(p.numel() for p in sam.parameters() if p.requires_grad)
        self.num_params = self.num_params_gdino + self.num_params_sam
        print(f"Segmenter initialized with {self.num_params_gdino} parameters for GDino and {self.num_params_sam} for SAM2, total: {self.num_params} parameters.")
        
    def get_params(self):
        return self.num_params  

    def segment(self, image, menu_text="food."):
        # grounding dino detection
        
        self.sam_pred.set_image(np.array(image))
        inputs = self.processor(images=image, text=menu_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.gdino(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=0.25, text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )
        r = results[0]
        boxes = r["boxes"].cpu().numpy()
        scores = r["scores"].cpu().numpy()
        names = r["labels"]
        # sam2 masks
        masks, _, _ = self.sam_pred.predict(None, None, box=boxes, multimask_output=False)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        masks = masks.astype(bool)
        return boxes, masks, names, scores
