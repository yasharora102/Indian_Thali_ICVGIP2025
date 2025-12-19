# mmseg/core/evaluation/metrics/iou_nan_absent.py

import numpy as np
import torch
from collections import OrderedDict
from prettytable import PrettyTable
from mmseg.registry import METRICS
from mmseg.evaluation.metrics import IoUMetric
from mmengine.logging import MMLogger, print_log
# from mmseg.utils import get_root_logger




@METRICS.register_module()
class IoUNanAbsent(IoUMetric):
    """IoU + accuracy metric that forces NaN for absent classes."""

    def compute_metrics(self, results: list) -> dict:
        """
        Override IoUMetric:
         - Mark IoU/Acc = NaN if union==0 (no GT & no pred) OR total_area_label==0.
         - Print the per-class table including those NaNs.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # 1) aggregate intersections/unions/pred-labels/gt-labels
        results = tuple(zip(*results))
        assert len(results) == 4
        total_area_intersect   = sum(results[0])
        total_area_union       = sum(results[1])
        total_area_pred_label  = sum(results[2])
        total_area_label       = sum(results[3])

        # 2) base metrics (per-class arrays in ret_metrics)
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )

        # 3) mask out absent classes → NaN for IoU and Acc
        for idx in range(len(total_area_union)):
            if total_area_union[idx] == 0 or total_area_label[idx] == 0:
                ret_metrics['IoU'][idx] = np.nan
                ret_metrics['Acc'][idx] = np.nan

        # 4) prepare summary (mean over classes, ignoring NaNs)
        #    ret_metrics keys: 'aAcc', 'IoU', 'Acc', etc.
        ret_metrics.pop('aAcc', None)
        summary = OrderedDict()
        for key, arr in ret_metrics.items():
            summary[key] = np.round(np.nanmean(arr) * 100, 2)

        # 5) build and print the per-class table
        class_names = self.dataset_meta['classes']
        per_class = OrderedDict([('Class', class_names)])
        for key, arr in ret_metrics.items():
            # multiply by 100 and round
            per_class[key] = np.round(arr * 100, 2).tolist()

        table = PrettyTable()
        for col, vals in per_class.items():
            table.add_column(col, vals)

        print_log('Per-class results:', logger)
        print_log('\n' + table.get_string(), logger=logger)

        # 6) return final summary metrics with standard keys
        final_metrics = {}
        for key, val in summary.items():
            if key == 'IoU':
                final_metrics['mIoU'] = val
            elif key == 'Acc':
                final_metrics['mAcc'] = val
            else:
                final_metrics['m' + key] = val

        return final_metrics
