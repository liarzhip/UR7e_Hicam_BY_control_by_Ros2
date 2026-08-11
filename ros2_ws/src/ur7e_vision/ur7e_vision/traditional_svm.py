# -*- coding: utf-8 -*-
"""
传统视觉 SVM 分类公共模块。

本模块不依赖海康 MVS，只依赖：
    pip install numpy opencv-python

它负责：
1. 根据轮廓裁剪并摆正目标；
2. 提取 HOG、Hu 矩和几何特征；
3. 保存训练样本；
4. 加载 OpenCV SVM 模型并预测类别。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ============================================================
# 特征配置：训练和实时检测必须使用完全相同的配置
# ============================================================

PATCH_SIZE = 128
PATCH_PADDING_RATIO = 0.12

HOG_WIN_SIZE = (PATCH_SIZE, PATCH_SIZE)
HOG_BLOCK_SIZE = (32, 32)
HOG_BLOCK_STRIDE = (16, 16)
HOG_CELL_SIZE = (16, 16)
HOG_BINS = 9

GEOMETRIC_FEATURE_NAMES = [
    "area_ratio",
    "aspect_ratio",
    "perimeter_normalized",
    "circularity",
    "rectangularity",
    "solidity",
    "extent",
    "vertex_count_normalized",
    "gray_mean",
    "gray_std",
    "edge_density",
]

HU_FEATURE_NAMES = [f"hu_{index + 1}" for index in range(7)]
EXTRA_FEATURE_NAMES = GEOMETRIC_FEATURE_NAMES + HU_FEATURE_NAMES

FEATURE_SIGNATURE = {
    "patch_size": PATCH_SIZE,
    "patch_padding_ratio": PATCH_PADDING_RATIO,
    "hog_win_size": list(HOG_WIN_SIZE),
    "hog_block_size": list(HOG_BLOCK_SIZE),
    "hog_block_stride": list(HOG_BLOCK_STRIDE),
    "hog_cell_size": list(HOG_CELL_SIZE),
    "hog_bins": HOG_BINS,
    "extra_feature_names": EXTRA_FEATURE_NAMES,
}


def create_hog_descriptor() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        _winSize=HOG_WIN_SIZE,
        _blockSize=HOG_BLOCK_SIZE,
        _blockStride=HOG_BLOCK_STRIDE,
        _cellSize=HOG_CELL_SIZE,
        _nbins=HOG_BINS,
    )


_HOG = create_hog_descriptor()


def get_feature_dimension() -> int:
    return int(_HOG.getDescriptorSize()) + len(EXTRA_FEATURE_NAMES)


# ============================================================
# 目标裁剪与特征提取
# ============================================================

def _order_quad_points(points: np.ndarray) -> np.ndarray:
    """将四边形点排序为：左上、右上、右下、左下。"""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)

    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]

    return ordered


def extract_rotated_patch(
    frame: np.ndarray,
    contour: np.ndarray,
    patch_size: int = PATCH_SIZE,
    padding_ratio: float = PATCH_PADDING_RATIO,
) -> np.ndarray:
    """
    根据轮廓的最小外接旋转矩形，透视裁剪并将目标长边统一到水平方向。
    """

    if frame is None or frame.size == 0:
        raise ValueError("frame 为空")

    if contour is None or len(contour) < 4:
        raise ValueError("contour 点数不足")

    rect = cv2.minAreaRect(contour)
    center, (width, height), angle = rect

    if width <= 1 or height <= 1:
        raise ValueError("目标旋转矩形尺寸无效")

    expanded_width = max(2.0, width * (1.0 + 2.0 * padding_ratio))
    expanded_height = max(2.0, height * (1.0 + 2.0 * padding_ratio))

    expanded_rect = (
        center,
        (expanded_width, expanded_height),
        angle,
    )

    source_points = _order_quad_points(
        cv2.boxPoints(expanded_rect)
    )

    top_width = np.linalg.norm(source_points[1] - source_points[0])
    bottom_width = np.linalg.norm(source_points[2] - source_points[3])
    left_height = np.linalg.norm(source_points[3] - source_points[0])
    right_height = np.linalg.norm(source_points[2] - source_points[1])

    crop_width = max(2, int(round(max(top_width, bottom_width))))
    crop_height = max(2, int(round(max(left_height, right_height))))

    destination_points = np.float32(
        [
            [0, 0],
            [crop_width - 1, 0],
            [crop_width - 1, crop_height - 1],
            [0, crop_height - 1],
        ]
    )

    perspective_matrix = cv2.getPerspectiveTransform(
        source_points,
        destination_points,
    )

    patch = cv2.warpPerspective(
        frame,
        perspective_matrix,
        (crop_width, crop_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # 将长边统一为水平方向，减少旋转对 HOG 的影响。
    if patch.shape[0] > patch.shape[1]:
        patch = cv2.rotate(
            patch,
            cv2.ROTATE_90_CLOCKWISE,
        )

    patch = cv2.resize(
        patch,
        (patch_size, patch_size),
        interpolation=cv2.INTER_AREA,
    )

    return patch


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else 0.0


def extract_shape_features(
    contour: np.ndarray,
    image_shape: tuple[int, ...],
    normalized_patch_gray: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """提取尺度归一化的轮廓几何特征、灰度统计和 Hu 矩。"""

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    if area <= 0 or perimeter <= 0:
        raise ValueError("轮廓面积或周长无效")

    (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
    long_side = max(float(rect_width), float(rect_height))
    short_side = max(1e-6, min(float(rect_width), float(rect_height)))
    rectangle_area = max(1e-6, long_side * short_side)

    hull = cv2.convexHull(contour)
    hull_area = max(1e-6, float(cv2.contourArea(hull)))

    bounding_x, bounding_y, bounding_width, bounding_height = cv2.boundingRect(contour)
    bounding_area = max(1.0, float(bounding_width * bounding_height))

    image_height, image_width = image_shape[:2]
    image_area = max(1.0, float(image_height * image_width))

    epsilon = 0.02 * perimeter
    polygon = cv2.approxPolyDP(
        contour,
        epsilon,
        True,
    )

    circularity = _safe_ratio(
        4.0 * math.pi * area,
        perimeter * perimeter,
    )

    gray_mean = float(np.mean(normalized_patch_gray)) / 255.0
    gray_std = float(np.std(normalized_patch_gray)) / 128.0

    patch_edges = cv2.Canny(
        normalized_patch_gray,
        60,
        160,
    )
    edge_density = float(np.count_nonzero(patch_edges)) / float(patch_edges.size)

    geometric_values = np.array(
        [
            area / image_area,
            min(long_side / short_side, 20.0),
            perimeter / math.sqrt(area),
            float(np.clip(circularity, 0.0, 1.5)),
            float(np.clip(area / rectangle_area, 0.0, 1.5)),
            float(np.clip(area / hull_area, 0.0, 1.5)),
            float(np.clip(area / bounding_area, 0.0, 1.5)),
            min(len(polygon), 30) / 30.0,
            gray_mean,
            gray_std,
            edge_density,
        ],
        dtype=np.float32,
    )

    moments = cv2.moments(contour)
    hu_values = cv2.HuMoments(moments).reshape(-1)

    # 对 Hu 矩做符号对数变换，并限制异常值范围。
    hu_log = (
        -np.sign(hu_values)
        * np.log10(np.abs(hu_values) + 1e-30)
    )
    hu_log = np.clip(
        hu_log,
        -20.0,
        20.0,
    ).astype(np.float32)

    feature_values = np.concatenate(
        [geometric_values, hu_log],
    ).astype(np.float32)

    feature_dict = {
        name: float(value)
        for name, value in zip(
            EXTRA_FEATURE_NAMES,
            feature_values,
            strict=True,
        )
    }

    return feature_values, feature_dict


def extract_feature_vector(
    frame: np.ndarray,
    contour: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    提取固定长度特征：
        HOG + 几何特征 + Hu 矩。

    返回：
        feature_vector：一维 float32 特征；
        patch：摆正后的 128×128 BGR 目标图；
        feature_dict：便于调试查看的特征字典。
    """

    patch = extract_rotated_patch(
        frame,
        contour,
    )

    if patch.ndim == 3:
        gray = cv2.cvtColor(
            patch,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = patch.copy()

    # 训练和推理都统一做直方图均衡，降低亮度变化影响。
    gray_normalized = cv2.equalizeHist(gray)

    hog_feature = _HOG.compute(
        gray_normalized,
    ).reshape(-1).astype(np.float32)

    shape_feature, feature_dict = extract_shape_features(
        contour,
        frame.shape,
        gray_normalized,
    )

    feature_vector = np.concatenate(
        [hog_feature, shape_feature],
    ).astype(np.float32)

    expected_dimension = get_feature_dimension()

    if feature_vector.size != expected_dimension:
        raise RuntimeError(
            f"特征维数异常：实际 {feature_vector.size}，"
            f"预期 {expected_dimension}"
        )

    return feature_vector, patch, feature_dict




def imwrite_unicode(
    file_path: str | Path,
    image: np.ndarray,
    params: list[int] | None = None,
) -> bool:
    """
    在 Windows 下兼容中文及其他 Unicode 路径保存图片。

    cv2.imwrite 在部分 OpenCV Windows 构建中无法处理非 ASCII 路径。
    这里先使用 cv2.imencode 编码到内存，再使用 NumPy.tofile 写入。
    """
    path = Path(file_path)

    if image is None or image.size == 0:
        return False

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extension = path.suffix.lower()

    if not extension:
        extension = ".png"

    try:
        success, encoded = cv2.imencode(
            extension,
            image,
            params or [],
        )

        if not success or encoded is None:
            return False

        encoded.tofile(
            str(path)
        )
    except (OSError, ValueError, cv2.error):
        return False

    return path.exists() and path.stat().st_size > 0


def _find_ascii_temp_root() -> Path:
    """
    寻找 OpenCV FileStorage 可使用的纯 ASCII 临时目录。

    OpenCV 的 SVM.save/SVM_load 在部分 Windows 构建中无法直接处理
    中文或其他 Unicode 路径，因此模型需要先在纯英文临时路径中
    保存或加载，再由 Python 复制到最终目录。
    """

    candidate_strings = [
        tempfile.gettempdir(),
        os.environ.get("TEMP", ""),
        os.environ.get("TMP", ""),
    ]

    system_drive = os.environ.get("SystemDrive", "C:")
    candidate_strings.append(
        str(Path(system_drive + "\\") / "opencv_svm_temp")
    )

    checked: set[str] = set()

    for candidate_string in candidate_strings:
        if not candidate_string:
            continue

        candidate = Path(candidate_string)

        normalized = str(candidate.resolve()) if candidate.exists() else str(candidate)

        if normalized in checked:
            continue

        checked.add(normalized)

        # OpenCV 接收的临时路径必须是纯 ASCII。
        if not str(candidate).isascii():
            continue

        try:
            candidate.mkdir(
                parents=True,
                exist_ok=True,
            )

            test_path = candidate / "opencv_svm_write_test.tmp"
            test_path.write_bytes(b"test")
            test_path.unlink(missing_ok=True)
        except OSError:
            continue

        return candidate

    raise RuntimeError(
        "没有找到可写的纯英文临时目录，无法兼容 OpenCV SVM 的中文路径限制。"
        "可以把项目临时移动到 C:\\HICAM，或设置 TEMP 为纯英文路径。"
    )


def save_svm_unicode(
    svm,
    target_path: str | Path,
) -> Path:
    """
    将 OpenCV SVM 模型保存到可能包含中文的最终路径。

    纯 ASCII 路径直接保存；Unicode 路径先保存到英文临时目录，
    再使用 Python shutil.copy2 复制到最终位置。
    """

    target = Path(target_path)
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        if str(target).isascii():
            svm.save(str(target))
        else:
            temp_root = _find_ascii_temp_root()

            with tempfile.TemporaryDirectory(
                prefix="opencv_svm_save_",
                dir=str(temp_root),
            ) as temp_directory:
                temp_model_path = (
                    Path(temp_directory)
                    / "svm_model.xml"
                )

                svm.save(
                    str(temp_model_path)
                )

                if (
                    not temp_model_path.exists()
                    or temp_model_path.stat().st_size <= 0
                ):
                    raise RuntimeError(
                        "OpenCV 没有成功生成临时 SVM 模型文件"
                    )

                shutil.copy2(
                    temp_model_path,
                    target,
                )
    except cv2.error as exc:
        raise RuntimeError(
            f"保存 SVM 模型失败：{target}\n{exc}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"复制 SVM 模型到最终路径失败：{target}\n{exc}"
        ) from exc

    if (
        not target.exists()
        or target.stat().st_size <= 0
    ):
        raise RuntimeError(
            f"SVM 模型文件保存后不存在或为空：{target}"
        )

    return target


def load_svm_unicode(
    model_path: str | Path,
):
    """
    从可能包含中文的路径加载 OpenCV SVM 模型。

    Unicode 路径先由 Python 复制到纯英文临时目录，再交给
    cv2.ml.SVM_load 读取。
    """

    source = Path(model_path)

    if not source.exists():
        raise FileNotFoundError(
            f"SVM 模型不存在：{source}"
        )

    try:
        if str(source).isascii():
            svm = cv2.ml.SVM_load(
                str(source)
            )
        else:
            temp_root = _find_ascii_temp_root()

            with tempfile.TemporaryDirectory(
                prefix="opencv_svm_load_",
                dir=str(temp_root),
            ) as temp_directory:
                temp_model_path = (
                    Path(temp_directory)
                    / "svm_model.xml"
                )

                shutil.copy2(
                    source,
                    temp_model_path,
                )

                svm = cv2.ml.SVM_load(
                    str(temp_model_path)
                )
    except cv2.error as exc:
        raise RuntimeError(
            f"加载 SVM 模型失败：{source}\n{exc}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"准备 SVM 临时模型文件失败：{source}\n{exc}"
        ) from exc

    if svm is None:
        raise RuntimeError(
            f"OpenCV 返回了空的 SVM 对象：{source}"
        )

    empty_method = getattr(
        svm,
        "empty",
        None,
    )

    if callable(empty_method) and empty_method():
        raise RuntimeError(
            f"加载到的 SVM 模型为空：{source}"
        )

    return svm


# ============================================================
# 训练样本保存
# ============================================================

def _sanitize_label(label: str) -> str:
    label = label.strip()

    if not label:
        raise ValueError("类别名称不能为空")

    if label in {".", ".."}:
        raise ValueError("类别名称无效")

    # 只替换路径分隔符和 Windows 不允许的字符；中文可正常保留。
    return re.sub(r'[\\/:*?"<>|]+', "_", label)


def save_feature_sample(
    dataset_dir: Path,
    label: str,
    feature_vector: np.ndarray,
    patch: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """
    保存一个训练样本：
        .npz：供训练脚本读取的特征；
        .png：供人工检查裁剪是否正确。
    """

    safe_label = _sanitize_label(label)
    class_dir = Path(dataset_dir) / safe_label
    class_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    npz_path = class_dir / f"{timestamp}.npz"
    image_path = class_dir / f"{timestamp}.png"

    metadata_payload = metadata or {}
    metadata_payload = {
        **metadata_payload,
        "label": safe_label,
        "feature_dimension": int(feature_vector.size),
        "feature_signature": FEATURE_SIGNATURE,
    }

    np.savez_compressed(
        npz_path,
        feature=np.asarray(feature_vector, dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps(
                metadata_payload,
                ensure_ascii=False,
            )
        ),
    )

    if not imwrite_unicode(
        image_path,
        patch,
    ):
        npz_path.unlink(missing_ok=True)
        raise RuntimeError(f"保存训练图像失败：{image_path}")

    return npz_path, image_path


# ============================================================
# SVM 模型加载与预测
# ============================================================

class SVMClassifier:
    """
    OpenCV 多分类 SVM 推理器。

    similarity_score 是样本与预测类别训练中心的相似度，
    不是概率。它用于拒绝明显超出训练分布的未知目标。
    """

    def __init__(
        self,
        model_dir: Path,
        unknown_score_threshold: float = 0.15,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.unknown_score_threshold = float(unknown_score_threshold)

        model_path = self.model_dir / "svm_model.xml"
        scaler_path = self.model_dir / "svm_scaler.npz"
        labels_path = self.model_dir / "labels.json"
        reference_path = self.model_dir / "class_reference.npz"
        metadata_path = self.model_dir / "model_meta.json"

        required_paths = [
            model_path,
            scaler_path,
            labels_path,
            reference_path,
            metadata_path,
        ]

        missing_paths = [
            path
            for path in required_paths
            if not path.exists()
        ]

        if missing_paths:
            missing_text = "\n".join(
                str(path)
                for path in missing_paths
            )
            raise FileNotFoundError(
                "SVM 模型文件不完整，缺少：\n"
                f"{missing_text}"
            )

        self.svm = load_svm_unicode(
            model_path
        )

        scaler_data = np.load(
            scaler_path,
            allow_pickle=False,
        )
        self.mean = scaler_data["mean"].astype(np.float32)
        self.std = scaler_data["std"].astype(np.float32)

        with labels_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            labels_data = json.load(file)

        self.id_to_label = {
            int(class_id): str(class_name)
            for class_id, class_name in labels_data.items()
        }

        reference_data = np.load(
            reference_path,
            allow_pickle=False,
        )
        self.class_ids = reference_data["class_ids"].astype(np.int32)
        self.class_centers = reference_data["centers"].astype(np.float32)
        self.class_radii = reference_data["radii"].astype(np.float32)

        self.reference_index = {
            int(class_id): index
            for index, class_id in enumerate(self.class_ids)
        }

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        expected_dimension = get_feature_dimension()
        model_dimension = int(self.metadata["feature_dimension"])

        if model_dimension != expected_dimension:
            raise RuntimeError(
                "当前程序的特征配置与已训练模型不一致："
                f"程序={expected_dimension}，模型={model_dimension}。"
                "请重新采集样本并训练模型。"
            )

        if self.mean.size != expected_dimension or self.std.size != expected_dimension:
            raise RuntimeError("标准化参数维数与当前特征维数不一致")

    def _scale(
        self,
        feature_vector: np.ndarray,
    ) -> np.ndarray:
        feature_vector = np.asarray(
            feature_vector,
            dtype=np.float32,
        ).reshape(-1)

        if feature_vector.size != self.mean.size:
            raise ValueError(
                f"预测特征维数为 {feature_vector.size}，"
                f"模型要求 {self.mean.size}"
            )

        return (
            (feature_vector - self.mean)
            / self.std
        ).astype(np.float32)

    def predict(
        self,
        feature_vector: np.ndarray,
    ) -> tuple[str, float, int]:
        scaled = self._scale(
            feature_vector
        )

        _, prediction = self.svm.predict(
            scaled.reshape(1, -1)
        )

        class_id = int(
            round(float(prediction[0, 0]))
        )

        class_name = self.id_to_label.get(
            class_id,
            f"class_{class_id}",
        )

        score = 1.0

        if class_id in self.reference_index:
            reference_index = self.reference_index[class_id]
            center = self.class_centers[reference_index]
            radius = max(
                float(self.class_radii[reference_index]),
                1e-6,
            )

            normalized_distance = (
                float(np.linalg.norm(scaled - center))
                / math.sqrt(float(scaled.size))
            )

            distance_ratio = normalized_distance / radius
            score = float(
                math.exp(
                    -0.5 * distance_ratio * distance_ratio
                )
            )

        if score < self.unknown_score_threshold:
            return "unknown", score, class_id

        return class_name, score, class_id
