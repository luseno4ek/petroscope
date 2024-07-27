from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from petroscope.segmentation.metrics import (
    SegmMetrics,
    acc,
    iou_per_class,
    to_hard,
)

from petroscope.segmentation.utils.data import (
    ClassAssociation,
    load_image,
    load_mask,
    to_categorical,
    void_borders,
)
from petroscope.segmentation.utils.vis import Plotter, SegmVisualizer


class SegmEvaluator:

    def __init__(self, codes_to_labels) -> None:
        self.codes_to_lbls = codes_to_labels
        self.buffer: list[SegmMetrics] = []

    def evaluate(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        void_mask: np.ndarray = None,
        add_to_buffer=True,
    ) -> SegmMetrics:
        assert pred.ndim >= 2

        # check if prediction is flat transform it to categorical
        if pred.ndim == 2:
            pred = to_categorical(pred, gt.shape[-1])

        # create a hard version of the prediction
        pred_hard = to_hard(pred)

        if void_mask is not None:
            assert void_mask.shape[:2] == gt.shape[:2]
            void = (
                np.repeat(void_mask[..., np.newaxis], gt.shape[-1], axis=-1)
                if void_mask.ndim == 2
                else void_mask
            )
            pred *= void
            pred_hard *= void
            gt *= void

        iou_class_soft = iou_per_class(gt, pred, self.codes_to_lbls)
        iou_class_hard = iou_per_class(gt, pred_hard, self.codes_to_lbls)

        img_metrics = SegmMetrics(
            iou_soft=iou_class_soft,
            iou=iou_class_hard,
            acc=acc(gt, pred_hard),
        )
        if add_to_buffer:
            self.buffer.append(img_metrics)
        return img_metrics

    def flush(self) -> SegmMetrics:
        ds_metrics = SegmMetrics.reduce(self.buffer)
        self.buffer.clear()
        return ds_metrics


class SegmDetailedTester:

    def __init__(
        self,
        out_dir: Path,
        classes: ClassAssociation,
        void_pad: int = 0,
        void_border_width: int = 0,
        vis_segmentation: bool = True,
        vis_plots: bool = True,
        log: bool = True,
    ):
        self.vis_segmentation = vis_segmentation
        self.vis_plots = vis_plots
        self.log = log
        self.out_dir = out_dir
        self.classes = classes
        self.void_w = void_border_width
        self.void_pad = void_pad
        self.metrics_history: list[SegmMetrics] = []
        self.metrics_void_history: list[SegmMetrics] = []
        self.eval = SegmEvaluator(codes_to_labels=classes.codes_to_labels)
        self.eval_void = SegmEvaluator(codes_to_labels=classes.codes_to_labels)

    def _visualize(
        self,
        img: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        void_mask: np.ndarray | None,
        out_dir: Path,
        img_name: str,
        void_color: tuple[int, int, int] = (0, 0, 255),
    ) -> None:
        assert img is not None
        img = (img * 255).astype(np.uint8)

        pred = (
            pred_mask if pred_mask.ndim == 2 else np.argmax(pred_mask, axis=-1)
        ).astype(np.uint8)
        gt = (
            gt_mask if gt_mask.ndim == 2 else np.argmax(gt_mask, axis=-1)
        ).astype(np.uint8)

        # visualize errors
        correct = (pred == gt).astype(np.uint8)
        if void_mask is not None:
            correct[void_mask == 0] = 255

        correct_v = SegmVisualizer.colorize_mask(
            correct,
            codes_to_colors={
                0: (244, 67, 54),
                1: (76, 175, 80),
                255: void_color,
            },
            return_image=True,
        )
        correct_colorized_mask = SegmVisualizer.colorize_mask(
            correct,
            codes_to_colors={0: (244, 67, 54)},
        )
        correct_v_overlay = SegmVisualizer.overlay(
            correct_colorized_mask, overlay=img
        )
        correct_v.save(out_dir / f"{img_name}_err.jpg")
        correct_v_overlay.save(out_dir / f"{img_name}_err_overlay.jpg")

        # visualize prediction
        pred_colorized = SegmVisualizer.colorize_mask(
            pred, self.classes.codes_to_colors
        )
        pred_v = SegmVisualizer.to_image(pred_colorized)
        pred_v_overlay = SegmVisualizer.overlay(pred_colorized, overlay=img)
        pred_v.save(out_dir / f"{img_name}_pred.jpg")
        pred_v_overlay.save(out_dir / f"{img_name}_pred_overlay.jpg")

    def _log(self, message: str, detailed: bool) -> None:
        log_name = "metrics_per_image.txt" if detailed else "metrics.txt"
        log = open(self.out_dir / log_name, "a+")
        log.write(message + "\n")
        log.close()

    def test_on_set(
        self,
        img_mask_paths: Iterable[tuple[Path, Path]],
        predict_func,
        description: str,
        return_void: bool = True,
    ) -> SegmMetrics:
        sub_dir = self.out_dir / description
        sub_dir.mkdir(exist_ok=True, parents=True)

        # iterate over all images in the set
        for img_mask_path in tqdm(img_mask_paths, "testing"):
            name = img_mask_path[0].stem
            img = load_image(img_mask_path[0], normalize=True)
            mask = load_mask(
                img_mask_path[1],
                classes=self.classes,
                one_hot=True,
            )
            pred = predict_func(img)
            void = void_borders(
                mask, border_width=self.void_w, pad=self.void_pad
            )
            # evaluate each image prediction (normal and void)
            metrics = self.eval.evaluate(pred, gt=mask)
            metrics_void = self.eval_void.evaluate(
                pred, gt=mask, void_mask=void
            )
            # log image metrics
            self._log(
                f"{description}, {name}:\n{metrics}\n"
                + f"{description}, {name} (void):\n{metrics_void}\n",
                detailed=True,
            )
            # visualize segmentation for image
            if self.vis_segmentation:
                self._visualize(img, mask, pred, void, sub_dir, f"img_{name}")

        # get total metrics for the set
        metrics_set = self.eval.flush()
        metrics_void_set = self.eval_void.flush()

        # save metrics to history (needed for plots)
        self.metrics_history.append(metrics_set)
        self.metrics_void_history.append(metrics_void_set)

        # log the total metrics
        log_str = (
            f"{description}, total:\n{metrics_set}\n"
            + f"{description}, total (void):\n{metrics_void_set}\n"
        )
        self._log(log_str, detailed=True)
        self._log(log_str, detailed=False)

        # draw plots
        if self.vis_plots:
            Plotter.plot_segm_metrics(
                self.metrics_history,
                self.out_dir,
                colors=self.classes.labels_to_colors_plt,
            )
            Plotter.plot_segm_metrics(
                self.metrics_void_history,
                self.out_dir,
                colors=self.classes.labels_to_colors_plt,
                name_suffix="_void",
            )

        # return resulted metrics
        if return_void:
            return metrics_set, metrics_void_set
        return metrics_set
