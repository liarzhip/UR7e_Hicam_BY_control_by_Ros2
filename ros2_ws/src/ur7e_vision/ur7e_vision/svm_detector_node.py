#!/usr/bin/env python3
import math
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

# IMPORTANT:
# Training and ROS inference now share EXACTLY the same feature extractor,
# scaler and classifier implementation.
from ur7e_vision.traditional_svm import (
    SVMClassifier,
    extract_feature_vector,
    get_feature_dimension,
)


def normalize_rect_angle(rect): # 归一化角度
    """
    Return an approximately [-90, 90) in-plane angle.
    OpenCV minAreaRect angle convention changes with rectangle side ordering.
    """
    (_, _), (w, h), angle = rect
    if w < h:
        angle += 90.0
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return float(angle)


def quaternion_normalize(q): # 
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Quaternion norm is zero.")
    return q / n


def quaternion_multiply(q1, q2):
    """Quaternion order: [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64)


def yaw_quaternion(yaw_rad):
    s = math.sin(yaw_rad / 2.0)
    c = math.cos(yaw_rad / 2.0)
    return np.array([0.0, 0.0, s, c], dtype=np.float64)


class SvmDetectorNode(Node):
    def __init__(self):
        super().__init__("svm_detector_node")

        share = Path(get_package_share_directory("ur7e_vision"))

        # ============================================================
        # ROS topics
        # ============================================================
        self.declare_parameter("image_topic", "/hik_camera/image_raw") # 订阅的图像话题
        self.declare_parameter("debug_topic", "/vision/debug_image") # 发布的调试图像话题
        self.declare_parameter("target_pose_topic", "/vision/target_pose") # 发布的目标位姿话题
        self.declare_parameter("target_class_topic", "/vision/target_class") # 发布的逐帧目标类别话题
        # 每次 5 帧定位周期只发布一次最终状态：
        #   target:bolt / target:nut / none / invalid
        self.declare_parameter("result_status_topic", "/vision/result_status")

        # ============================================================
        # SVM model parameters
        # ============================================================
        # Keep the old svm_model parameter for backward compatibility.
        # The original classifier needs FIVE files in the same directory:
        #   svm_model.xml
        #   svm_scaler.npz
        #   labels.json
        #   class_reference.npz
        #   model_meta.json
        self.declare_parameter(
            "svm_model",
            str(share / "models" / "svm_model.xml"),
        )

        # Optional explicit model directory.
        # If empty, the parent directory of svm_model is used.
        self.declare_parameter("svm_model_dir", str(share / "models"))
        self.declare_parameter("unknown_class_score_threshold", 0.15)

        # Keep labels_file as a compatibility/fallback interface.
        # When the complete SVMClassifier is loaded, labels.json from the
        # model directory is the authoritative class-name mapping.
        self.declare_parameter(
            "labels_file",
            str(share / "config" / "labels.yaml"),
        )

        self.declare_parameter(
            "homography_file",
            str(share / "calibration" / "homography.yaml"),
        )

        # ============================================================
        # Contour preprocessing parameters
        # ============================================================
        self.declare_parameter("min_area", 500.0)
        self.declare_parameter("max_area", 10000000.0)
        self.declare_parameter("blur_size", 5)
        self.declare_parameter("threshold_mode", "otsu")  # otsu | binary | canny
        self.declare_parameter("binary_threshold", 100)
        self.declare_parameter("canny_low", 50)
        self.declare_parameter("canny_high", 150)
        self.declare_parameter("invert_binary", False)

        # ============================================================
        # Robot target parameters
        # ============================================================
        self.declare_parameter("target_z", 0.165)
        self.declare_parameter("target_frame", "base")

        # Homography -> Base XY 后的固定系统补偿，单位：m
        # 当前机械臂需要沿 Base -X、-Y 各补偿 10 mm。
        self.declare_parameter("target_offset_x", -0.010)
        self.declare_parameter("target_offset_y", -0.015)

        self.declare_parameter("fixed_qx", 0.995)
        self.declare_parameter("fixed_qy", 0.086)
        self.declare_parameter("fixed_qz", 0.019)
        self.declare_parameter("fixed_qw", -0.047)
        self.declare_parameter("use_detection_angle", False)
        self.declare_parameter("yaw_offset_deg", 0.0)

        self.declare_parameter("target_class_id", -1)
        self.declare_parameter("publish_when_no_svm", True)
        self.declare_parameter("stable_frames", 5)
        self.declare_parameter("stable_pixel_tolerance", 3.0)

        # ============================================================
        # Read parameters
        # ============================================================
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)
        self.target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self.target_class_topic = str(self.get_parameter("target_class_topic").value)
        self.result_status_topic = str(
            self.get_parameter("result_status_topic").value
        )

        svm_model_value = str(self.get_parameter("svm_model").value).strip()
        svm_model_dir_value = str(
            self.get_parameter("svm_model_dir").value
        ).strip()
        labels_value = str(self.get_parameter("labels_file").value).strip()
        homography_value = str(
            self.get_parameter("homography_file").value
        ).strip()

        self.svm_model_path = (
            Path(svm_model_value) if svm_model_value else None
        )

        if svm_model_dir_value:
            self.svm_model_dir = Path(svm_model_dir_value)
        elif self.svm_model_path is not None:
            self.svm_model_dir = self.svm_model_path.parent
        else:
            self.svm_model_dir = None

        self.unknown_class_score_threshold = float(
            self.get_parameter("unknown_class_score_threshold").value
        )

        self.labels_path = Path(labels_value) if labels_value else None
        self.homography_path = (
            Path(homography_value) if homography_value else None
        )

        self.min_area = float(self.get_parameter("min_area").value)
        self.max_area = float(self.get_parameter("max_area").value)
        self.blur_size = int(self.get_parameter("blur_size").value)
        self.threshold_mode = str(
            self.get_parameter("threshold_mode").value
        ).lower()
        self.binary_threshold = int(
            self.get_parameter("binary_threshold").value
        )
        self.canny_low = int(self.get_parameter("canny_low").value)
        self.canny_high = int(self.get_parameter("canny_high").value)
        self.invert_binary = bool(
            self.get_parameter("invert_binary").value
        )

        self.target_z = float(self.get_parameter("target_z").value)
        self.target_frame = str(self.get_parameter("target_frame").value)

        self.target_offset_x = float(
            self.get_parameter("target_offset_x").value
        )
        self.target_offset_y = float(
            self.get_parameter("target_offset_y").value
        )

        self.fixed_q = quaternion_normalize([
            float(self.get_parameter("fixed_qx").value),
            float(self.get_parameter("fixed_qy").value),
            float(self.get_parameter("fixed_qz").value),
            float(self.get_parameter("fixed_qw").value),
        ])
        self.use_detection_angle = bool(
            self.get_parameter("use_detection_angle").value
        )
        self.yaw_offset_deg = float(
            self.get_parameter("yaw_offset_deg").value
        )

        self.target_class_id = int(
            self.get_parameter("target_class_id").value
        )
        self.publish_when_no_svm = bool(
            self.get_parameter("publish_when_no_svm").value
        )
        self.stable_frames = max(
            1,
            int(self.get_parameter("stable_frames").value),
        )
        self.stable_pixel_tolerance = float(
            self.get_parameter("stable_pixel_tolerance").value
        )

        # ============================================================
        # Runtime objects
        # ============================================================
        self.bridge = CvBridge()

        # Complete original SVM pipeline:
        # feature extraction -> scaler -> SVM -> labels/reference score.
        self.classifier = self._load_classifier()

        # Compatibility fallback only when classifier is unavailable.
        self.labels = self._load_labels()

        self.H = self._load_homography()

        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.target_pose_topic,
            10,
        )
        self.class_pub = self.create_publisher(
            String,
            self.target_class_topic,
            10,
        )
        self.result_pub = self.create_publisher(
            String,
            self.result_status_topic,
            10,
        )
        self.debug_pub = self.create_publisher(
            Image,
            self.debug_topic,
            5,
        )
        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.on_image,
            5,
        )

        self.stable_history = []

        # 一次 /hik_camera/run_once 会产生 stable_frames 张图。
        # 这里对整轮图像计数，从而可以明确区分：
        #   none    = 整轮都没有候选目标
        #   invalid = 看到了候选，但本轮没有形成合法 bolt/nut 位姿
        self.cycle_frame_count = 0
        self.cycle_had_candidate = False
        self.class_history = []

        self.get_logger().info(
            f"Detector ready. image={self.image_topic}, "
            f"target={self.target_pose_topic}, "
            f"result={self.result_status_topic}"
        )
        self.get_logger().info(
            "Base XY compensation: "
            f"dX={self.target_offset_x * 1000.0:.1f} mm, "
            f"dY={self.target_offset_y * 1000.0:.1f} mm"
        )

        if self.classifier is not None:
            self.get_logger().info(
                "SVM pipeline ready: "
                f"feature_dimension={get_feature_dimension()}, "
                f"model_dir={self.svm_model_dir}"
            )

        if self.H is None:
            self.get_logger().warning(
                "No valid homography loaded. Detection/SVM classification "
                "will work, but target_pose will NOT be published until "
                "planar calibration is completed."
            )

        if self.target_z < -10.0:
            self.get_logger().warning(
                "target_z is not configured. target_pose publication is disabled."
            )

    # ================================================================
    # Model / calibration loading
    # ================================================================
    def _load_classifier(self):
        if self.svm_model_dir is None:
            self.get_logger().warning(
                "SVM model directory is not configured. "
                "Running contour-only mode."
            )
            return None

        try:
            classifier = SVMClassifier(
                self.svm_model_dir,
                unknown_score_threshold=(
                    self.unknown_class_score_threshold
                ),
            )
        except Exception as exc:
            self.get_logger().error(
                "Failed to load complete SVM pipeline from "
                f"{self.svm_model_dir}: {exc}"
            )
            self.get_logger().warning(
                "SVM classification is disabled. "
                "Make sure svm_model.xml, svm_scaler.npz, labels.json, "
                "class_reference.npz and model_meta.json are all in the "
                "same model directory."
            )
            return None

        model_feature_count = int(classifier.svm.getVarCount())
        extractor_feature_count = int(get_feature_dimension())

        if model_feature_count != extractor_feature_count:
            raise RuntimeError(
                "SVM feature dimension mismatch even after loading the "
                "original training pipeline: "
                f"model={model_feature_count}, "
                f"extractor={extractor_feature_count}."
            )

        self.get_logger().info(
            "Loaded complete SVM pipeline: "
            f"{self.svm_model_dir}, "
            f"features={model_feature_count}"
        )
        return classifier

    def _load_labels(self):
        if self.labels_path is None or not self.labels_path.is_file():
            return {}

        with open(
            self.labels_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = yaml.safe_load(f) or {}

        raw = data.get("labels", {})
        return {
            int(k): str(v)
            for k, v in raw.items()
        }

    def _load_homography(self):
        if (
            self.homography_path is None
            or not self.homography_path.is_file()
        ):
            return None

        with open(
            self.homography_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = yaml.safe_load(f) or {}

        values = data.get("homography", {}).get("data", None)

        if values is None or len(values) != 9:
            self.get_logger().error(
                f"Invalid homography file: {self.homography_path}"
            )
            return None

        H = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(3, 3)

        self.get_logger().info(
            f"Loaded homography: {self.homography_path}"
        )
        return H

    # ================================================================
    # Image preprocessing / SVM
    # ================================================================
    def preprocess(self, gray):
        k = self.blur_size

        if k < 1:
            k = 1

        if k % 2 == 0:
            k += 1

        blurred = cv2.GaussianBlur(
            gray,
            (k, k),
            0,
        )

        if self.threshold_mode == "canny":
            return cv2.Canny(
                blurred,
                self.canny_low,
                self.canny_high,
            )

        if self.threshold_mode == "binary":
            flag = (
                cv2.THRESH_BINARY_INV
                if self.invert_binary
                else cv2.THRESH_BINARY
            )
            _, mask = cv2.threshold(
                blurred,
                self.binary_threshold,
                255,
                flag,
            )
            return mask

        flag = (
            cv2.THRESH_BINARY_INV
            if self.invert_binary
            else cv2.THRESH_BINARY
        )

        _, mask = cv2.threshold(
            blurred,
            0,
            255,
            flag | cv2.THRESH_OTSU,
        )

        return mask

    def classify(self, frame, contour):
        """
        Run the EXACT feature pipeline used by the original training code.

        Returns:
            (class_name, class_score, class_id)

        If the SVM is unavailable:
            ("unclassified", 0.0, -1)
        """
        if self.classifier is None:
            return "unclassified", 0.0, -1

        feature_vector, normalized_patch, feature_details = (
            extract_feature_vector(
                frame,
                contour,
            )
        )

        expected = int(get_feature_dimension())

        if feature_vector.size != expected:
            raise RuntimeError(
                "Feature extractor returned unexpected dimension: "
                f"{feature_vector.size}, expected {expected}."
            )

        class_name, class_score, class_id = (
            self.classifier.predict(
                feature_vector
            )
        )

        return (
            str(class_name),
            float(class_score),
            int(class_id),
        )

    # ================================================================
    # Coordinate / stability helpers
    # ================================================================
    def pixel_to_base_xy(self, u, v):
        src = np.array(
            [[[float(u), float(v)]]],
            dtype=np.float64,
        )
        dst = cv2.perspectiveTransform(
            src,
            self.H,
        )
        # Homography 得到原始 Base XY 后，再叠加固定系统补偿。
        # 这样 Homography 只负责坐标映射，offset 负责 TCP/夹爪等固定偏差。
        base_x = float(dst[0, 0, 0]) + self.target_offset_x
        base_y = float(dst[0, 0, 1]) + self.target_offset_y

        return (
            base_x,
            base_y,
        )

    def is_stable(self, u, v):
        """
        收集连续 stable_frames 帧目标中心，并计算平均坐标。

        返回：
            ready:
                是否已经收集满 stable_frames 帧。

            stable:
                True  = 所有样本相对平均中心的最大偏差
                        <= stable_pixel_tolerance。
                False = 已收集满，但波动超过阈值。

            center:
                stable_frames 帧的平均像素坐标 [u_mean, v_mean]。
                未收集满时返回 None。

            max_dev:
                stable_frames 帧中距离平均中心最大的像素距离。
                未收集满时返回 None。

        注意：
            stable=False 不再阻止最终位姿发布。
            当收集满 stable_frames 帧以后，无论 stable True/False，
            都会使用平均坐标进行后续定位。
        """
        self.stable_history.append(
            np.array(
                [u, v],
                dtype=np.float64,
            )
        )

        # 正常情况下 HIK 节点每轮正好发布 stable_frames 帧。
        # 这里仍保留滑动窗口保护。
        if len(self.stable_history) > self.stable_frames:
            self.stable_history.pop(0)

        # 尚未收集满规定帧数，不输出最终定位结果。
        if len(self.stable_history) < self.stable_frames:
            return False, False, None, None

        arr = np.vstack(
            self.stable_history
        )

        # ----------------------------------------------------
        # 最终用于机械臂定位的 5 帧平均像素坐标
        # ----------------------------------------------------
        center = arr.mean(
            axis=0
        )

        # 每一帧相对平均中心的二维欧氏距离。
        deviations = np.linalg.norm(
            arr - center,
            axis=1,
        )

        max_dev = float(
            deviations.max()
        )

        stable = bool(
            max_dev
            <= self.stable_pixel_tolerance
        )

        return (
            True,
            stable,
            center,
            max_dev,
        )



    def _publish_result_status(self, status):
        msg = String()
        msg.data = str(status)
        self.result_pub.publish(msg)
        self.get_logger().info(
            f"Vision cycle result: {msg.data}"
        )

    def _reset_positioning_cycle(self):
        self.stable_history.clear()
        self.class_history.clear()
        self.cycle_frame_count = 0
        self.cycle_had_candidate = False

    def _finish_positioning_cycle(self, status):
        self._publish_result_status(status)
        self._reset_positioning_cycle()

    def _final_cycle_label(self):
        """
        对本轮已经选中的目标类别做多数投票。
        只把 bolt / nut 视为可分拣类别。
        """
        valid = [
            str(label).strip().lower()
            for label in self.class_history
            if str(label).strip().lower() in ("bolt", "nut")
        ]

        if not valid:
            return None

        bolt_count = valid.count("bolt")
        nut_count = valid.count("nut")

        if bolt_count == nut_count:
            return None

        return "bolt" if bolt_count > nut_count else "nut"

    # ================================================================
    # Main image callback
    # ================================================================
    def on_image(self, msg):
        try:
            # HIK 当前每次 run_once 发布 stable_frames 张图。
            # 无论这一帧有没有检测到目标，都属于当前定位周期。
            self.cycle_frame_count += 1

            img = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="passthrough",
            )

            if img.ndim == 2:
                gray = img
                debug = cv2.cvtColor(
                    gray,
                    cv2.COLOR_GRAY2BGR,
                )
            else:
                debug = img.copy()
                gray = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2GRAY,
                )

            mask = self.preprocess(
                gray
            )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            candidates = []

            for contour in contours:
                area = float(
                    cv2.contourArea(contour)
                )

                if (
                    area < self.min_area
                    or area > self.max_area
                ):
                    continue

                rect = cv2.minAreaRect(
                    contour
                )
                (u, v), (rw, rh), _ = rect

                if rw <= 1.0 or rh <= 1.0:
                    continue

                try:
                    (
                        class_name,
                        class_score,
                        class_id,
                    ) = self.classify(
                        img,
                        contour,
                    )
                except Exception as exc:
                    self.get_logger().error(
                        f"SVM classification failed: {exc}"
                    )
                    continue

                if (
                    self.classifier is not None
                    and self.target_class_id >= 0
                    and class_id != self.target_class_id
                ):
                    continue

                if (
                    self.classifier is None
                    and not self.publish_when_no_svm
                ):
                    continue

                angle_deg = normalize_rect_angle(
                    rect
                )

                candidates.append(
                    (
                        area,
                        contour,
                        u,
                        v,
                        angle_deg,
                        class_name,
                        class_score,
                        class_id,
                    )
                )

            if not candidates:
                self._publish_debug(
                    msg,
                    debug,
                )

                # 只有整轮 stable_frames 张图都没有候选，才认为桌面为空。
                # 如果前面某些帧看到了候选而本轮最终不完整，则返回 invalid，
                # 由任务节点重拍，而不是误认为分拣已经结束。
                if self.cycle_frame_count >= self.stable_frames:
                    status = (
                        "invalid"
                        if self.cycle_had_candidate
                        else "none"
                    )
                    self._finish_positioning_cycle(status)

                return

            # Preserve the original ROS V1 policy:
            # choose the largest accepted contour.
            candidates.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            (
                area,
                contour,
                u,
                v,
                angle_deg,
                class_name,
                class_score,
                class_id,
            ) = candidates[0]

            box = cv2.boxPoints(
                cv2.minAreaRect(contour)
            ).astype(np.int32)

            cv2.drawContours(
                debug,
                [box],
                0,
                (0, 255, 0),
                2,
            )

            cv2.circle(
                debug,
                (
                    int(round(u)),
                    int(round(v)),
                ),
                5,
                (0, 0, 255),
                -1,
            )

            # When the complete classifier is available, labels.json from the
            # original model is authoritative.  labels.yaml is only a fallback
            # for contour-only mode / legacy use.
            if self.classifier is not None:
                label = class_name
            else:
                label = self.labels.get(
                    class_id,
                    f"class_{class_id}",
                )

            self.cycle_had_candidate = True
            self.class_history.append(
                str(label).strip().lower()
            )

            cv2.putText(
                debug,
                (
                    f"{label} "
                    f"score={class_score:.3f} "
                    f"u={u:.1f} v={v:.1f} "
                    f"a={angle_deg:.1f}"
                ),
                (
                    max(0, int(u) - 180),
                    max(25, int(v) - 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            (
                stable_ready,
                stable,
                stable_center,
                stable_max_dev,
            ) = self.is_stable(
                u,
                v,
            )

            # 前 stable_frames-1 帧只用于积累定位数据。
            if stable_ready:
                stable_u = float(
                    stable_center[0]
                )
                stable_v = float(
                    stable_center[1]
                )

                stability_text = (
                    f"stable={stable} "
                    f"avg=({stable_u:.1f},{stable_v:.1f}) "
                    f"max_dev={stable_max_dev:.2f}px"
                )
            else:
                stable_u = None
                stable_v = None

                stability_text = (
                    f"collecting="
                    f"{len(self.stable_history)}/"
                    f"{self.stable_frames}"
                )

            cv2.putText(
                debug,
                stability_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            class_msg = String()
            class_msg.data = label
            self.class_pub.publish(
                class_msg
            )

            final_label = (
                self._final_cycle_label()
                if stable_ready
                else None
            )

            # Homography 加载以后，还可以同时显示机器人 XY 和相机里面的 uv
            if self.H is not None:
                base_x, base_y = self.pixel_to_base_xy(u, v)

                self.get_logger().info(
                    f"Detected: class={label}, "
                    f"score={class_score:.3f}, "
                    f"pixel=({u:.1f}, {v:.1f}), "
                    f"base=({base_x:.4f}, {base_y:.4f}) m, "
                    f"angle={angle_deg:.1f} deg",
                    throttle_duration_sec=1.0,
                )
            else:
                self.get_logger().info(
                    f"Detected: class={label}, "
                    f"score={class_score:.3f}, "
                    f"pixel=({u:.1f}, {v:.1f}), "
                    f"angle={angle_deg:.1f} deg",
                    throttle_duration_sec=1.0,
                )

            # ========================================================
            # 5 帧最终定位结果
            # ========================================================
            # 与原版不同：
            #
            # stable=True
            #   -> 使用 5 帧平均坐标正常发布。
            #
            # stable=False
            #   -> 仍然使用 5 帧平均坐标发布，
            #      但通过 WARNING 明确告诉操作者当前定位波动较大。
            #
            # 尚未收集满 stable_frames 帧时，不发布最终 target_pose。
            #
            # 其他安全条件仍然保留：
            #   1. 必须已经完成 Homography 标定；
            #   2. target_z 必须有效；
            #   3. SVM 结果不能为 unknown。
            if (
                stable_ready
                and self.H is not None
                and self.target_z > -10.0
                and final_label in ("bolt", "nut")
            ):
                # ----------------------------------------------------
                # 关键修改：
                # 使用 stable_frames 帧的平均像素坐标，
                # 而不是当前第 5 帧的 (u, v)。
                # ----------------------------------------------------
                base_x, base_y = self.pixel_to_base_xy(
                    stable_u,
                    stable_v,
                )

                if stable:
                    self.get_logger().info(
                        f"Stable positioning result: "
                        f"class={final_label}, "
                        f"frames={self.stable_frames}, "
                        f"avg_pixel=({stable_u:.2f}, {stable_v:.2f}), "
                        f"max_dev={stable_max_dev:.2f}px <= "
                        f"{self.stable_pixel_tolerance:.2f}px, "
                        f"base=({base_x:.4f}, {base_y:.4f}) m"
                    )
                else:
                    self.get_logger().warning(
                        f"UNSTABLE positioning: "
                        f"class={final_label}, "
                        f"frames={self.stable_frames}, "
                        f"avg_pixel=({stable_u:.2f}, {stable_v:.2f}), "
                        f"max_dev={stable_max_dev:.2f}px > "
                        f"{self.stable_pixel_tolerance:.2f}px. "
                        f"Still publishing the AVERAGE position: "
                        f"base=({base_x:.4f}, {base_y:.4f}) m"
                    )

                q = self.fixed_q.copy()

                if self.use_detection_angle:
                    yaw = math.radians(
                        angle_deg
                        + self.yaw_offset_deg
                    )

                    q = quaternion_multiply(
                        yaw_quaternion(yaw),
                        q,
                    )

                    q = quaternion_normalize(
                        q
                    )

                pose = PoseStamped()
                pose.header.stamp = msg.header.stamp
                pose.header.frame_id = self.target_frame

                pose.pose.position.x = base_x
                pose.pose.position.y = base_y
                pose.pose.position.z = self.target_z

                pose.pose.orientation.x = float(q[0])
                pose.pose.orientation.y = float(q[1])
                pose.pose.orientation.z = float(q[2])
                pose.pose.orientation.w = float(q[3])

                self.pose_pub.publish(
                    pose
                )

                # 最终类别与最终 Pose 同一轮发布，任务节点据此冻结本轮目标。
                final_class_msg = String()
                final_class_msg.data = final_label
                self.class_pub.publish(final_class_msg)

                cv2.putText(
                    debug,
                    (
                        f"AVG base=({base_x:.3f},"
                        f"{base_y:.3f},"
                        f"{self.target_z:.3f})"
                    ),
                    (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # 本轮形成合法目标：
                # 先发布 Pose，再发布 target:<class> 最终状态，然后清空整轮缓存。
                self._finish_positioning_cycle(
                    f"target:{final_label}"
                )

            # 已经走完一整轮图像，但没有形成合法 bolt/nut 位姿。
            # 这不是“桌面为空”，而是 invalid；任务节点会按配置重新拍照。
            if (
                self.cycle_frame_count >= self.stable_frames
                and self.cycle_frame_count != 0
            ):
                self.get_logger().warning(
                    "Vision cycle completed but no valid bolt/nut pose was "
                    "produced. Publishing result=invalid."
                )
                self._finish_positioning_cycle("invalid")

            self._publish_debug(
                msg,
                debug,
            )

        except Exception as exc:
            self.get_logger().error(
                f"Detector callback failed: {exc}"
            )

    def _publish_debug(self, source_msg, debug):
        out = self.bridge.cv2_to_imgmsg(
            debug,
            encoding="bgr8",
        )
        out.header = source_msg.header
        self.debug_pub.publish(
            out
        )


def main(args=None):
    rclpy.init(args=args)
    node = SvmDetectorNode()

    try:
        rclpy.spin(
            node
        )
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
