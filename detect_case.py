from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.engine.trainer import BaseTrainer
import torch
from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode

class GetaTrainer(DetectionTrainer):
    def setup_model(self):
        # Call the base logic to build and move model
        ckpt = super().setup_model()

        # Now perform model‐modification (quant + prune setup)
        self.model = model_to_quantize_model(
            self.model,
            quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION,
            num_bits=16
        )

        dummy_input = torch.randn(1, 3, self.args.imgsz, self.args.imgsz)
        self.oto = OTO(model=self.model, dummy_input=dummy_input)

        self.oto.mark_unprunable_by_param_names([
            'model.22.dfl.conv.weight',
            'model.22.cv3.2.2.weight',
            'model.22.cv2.2.2.weight',
            'model.22.cv2.1.2.weight',
            'model.22.cv3.1.2.weight',
            'model.22.cv3.0.2.weight',
            'model.22.cv2.0.2.weight'
        ])

        for node_group in self.oto._graph.node_groups.values():
            for node in node_group:
                if node.op_name == 'slice':
                    node_group.is_prunable = False

        # self.oto.visualize(view=False, out_dir='.')
        return ckpt

    def build_optimizer(self, model, name="auto", lr=None, momentum=None, decay=None, iterations=None):
        # Use GETA to build optimizer based on the modified model
        optimizer = self.oto.geta(
            variant="sgd",
            lr=1e-3,
            lr_quant=1e-3,
            first_momentum=0.9,
            weight_decay=1e-4,
            target_group_sparsity=0.5,
            start_projection_step=5 * len(self.train_loader),
            projection_periods=5,
            projection_steps=2 * len(self.train_loader),
            start_pruning_step=10 * len(self.train_loader),
            pruning_periods=5,
            pruning_steps=10 * len(self.train_loader),
            bit_reduction=2,
            min_bit_wt=4,
            max_bit_wt=16
        )
        return optimizer

    def save_model(self):
        self.oto.construct_subnet(out_dir="./")

args = dict(model="yolov8n.pt", data="VOC.yaml", epochs=50, amp=False, batch=32, workers=16)
a = GetaTrainer(overrides=args)
a.train()