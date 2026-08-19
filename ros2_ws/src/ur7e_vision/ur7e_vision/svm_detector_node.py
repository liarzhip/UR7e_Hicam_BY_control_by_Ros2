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
        self.declare_parameter("target_offset_x", -0.005)
        self.declare_parameter("target_offset_y", -0.007)

        # Zero-angle grasp reference: the user's taught HOME pose with
        # the gripper level / vertical-down.
        self.declare_parameter("fixed_qx", 0.9238700051322959)
        self.declare_parameter("fixed_qy", 0.38269617379663273)
        self.declare_parameter("fixed_qz", -0.0022329597426112052)
        self.declare_parameter("fixed_qw", 0.0016929468558880931)

        # Master switch plus class-specific switches.
        # Bolt has a meaningful long-axis angle; nut is rotationally symmetric,
        # so nut angle is disabled by default.
        self.declare_parameter("use_detection_angle", True)
        self.declare_parameter("use_detection_angle_for_bolt", True)
        self.declare_parameter("use_detection_angle_for_nut", True)

        # Fine adjustment after image-angle -> Base yaw-delta conversion.
        self.declare_parameter("yaw_offset_deg", 0.0)

        self.declare_parameter("target_class_id", -1)
        self.declare_parameter("publish_when_no_svm", True)
        self.declare_parameter("stable_frames", 5)
        self.declare_parameter("stable_pixel_tolerance", 3.0)

        # ============================================================
        # Multi-target association / classification robustness
        # ============================================================
        # First frame: confidence | largest
        self.declare_parameter("target_selection_mode", "confidence")

        # Only bolt/nut predictions above this score are eligible to be picked.
        # Low-confidence/unknown objects make the cycle INVALID rather than NONE.
        self.declare_parameter("min_target_score", 0.55)

        # After the first frame locks one object, later frames may only match
        # a candidate near the previously tracked center.
        self.declare_parameter("target_match_radius_px", 80.0)

        # A 5-frame cycle may tolerate one missed frame, but still requires
        # enough observations of the SAME tracked target.
        self.declare_parameter("min_track_frames", 4)

        # Final class must have at least this many votes on the same track.
        self.declare_parameter("min_class_consistency_frames", 4)

        # Never publish a position if the tracked centers are not stable.
        self.declare_parameter("require_stable_position", True)

        # Pixel vector length used to transform image angle through Homography
        # into a Base-frame yaw.
        self.declare_parameter("angle_vector_length_px", 80.0)

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

        self.use_detection_angle_for_bolt = bool(
            self.get_parameter("use_detection_angle_for_bolt").value
        )

        self.use_detection_angle_for_nut = bool(
            self.get_parameter("use_detection_angle_for_nut").value
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

        self.target_selection_mode = str(
            self.get_parameter("target_selection_mode").value
        ).strip().lower()

        self.min_target_score = float(
            self.get_parameter("min_target_score").value
        )

        self.target_match_radius_px = float(
            self.get_parameter("target_match_radius_px").value
        )

        self.min_track_frames = max(
            1,
            int(self.get_parameter("min_track_frames").value),
        )

        self.min_class_consistency_frames = max(
            1,
            int(self.get_parameter("min_class_consistency_frames").value),
        )

        self.require_stable_position = bool(
            self.get_parameter("require_stable_position").value
        )

        self.angle_vector_length_px = max(
            5.0,
            float(self.get_parameter("angle_vector_length_px").value),
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

        # One run_once cycle tracks ONE physical object only.
        self.stable_history = []
        self.angle_history = []
        self.class_history = []
        self.score_history = []

        self.track_center = None
        self.cycle_frame_count = 0

        # True when an object-like contour was seen, even if SVM confidence was
        # too low. This prevents "classification failed" from being mistaken
        # for an empty table.
        self.cycle_had_object_like_contour = False

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

        self.get_logger().info(
            "Grasp reference orientation (angle=0 deg): "
            "q=(0.923870, 0.382696, -0.002233, 0.001693); "
            "bolt_angle=ON, nut_angle=OFF"
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

    def _publish_result_status(self, status):
        msg = String()
        msg.data = str(status)
        self.result_pub.publish(msg)
        self.get_logger().info(
            f"Vision cycle result: {msg.data}"
        )

    def _reset_positioning_cycle(self):
        self.stable_history.clear()
        self.angle_history.clear()
        self.class_history.clear()
        self.score_history.clear()

        self.track_center = None
        self.cycle_frame_count = 0
        self.cycle_had_object_like_contour = False

    def _finish_positioning_cycle(self, status):
        self._publish_result_status(status)
        self._reset_positioning_cycle()

    def _select_initial_candidate(self, candidates):
        """
        Lock ONE physical target on the first usable frame.

        confidence:
            choose the most confidently classified bolt/nut,
            then use area as a tie-breaker.

        largest:
            preserve the old policy.
        """
        if not candidates:
            return None

        if self.target_selection_mode == "largest":
            return max(
                candidates,
                key=lambda c: (
                    c["area"],
                    c["score"],
                ),
            )

        return max(
            candidates,
            key=lambda c: (
                c["score"],
                c["area"],
            ),
        )

    def _match_tracked_candidate(self, candidates):
        """
        Associate later frames with the SAME target by nearest-center gating.
        """
        if not candidates:
            return None

        if self.track_center is None:
            return self._select_initial_candidate(
                candidates
            )

        tx, ty = self.track_center

        ranked = []
        for candidate in candidates:
            d = math.hypot(
                candidate["u"] - tx,
                candidate["v"] - ty,
            )
            ranked.append(
                (d, -candidate["score"], -candidate["area"], candidate)
            )

        ranked.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            )
        )

        distance, _, _, selected = ranked[0]

        if distance > self.target_match_radius_px:
            return None

        return selected

    def _final_cycle_label(self):
        """
        Majority vote on the SAME tracked target.
        """
        valid = [
            str(label).strip().lower()
            for label in self.class_history
            if str(label).strip().lower() in ("bolt", "nut")
        ]

        if not valid:
            return None, 0

        bolt_count = valid.count("bolt")
        nut_count = valid.count("nut")

        if bolt_count == nut_count:
            return None, max(
                bolt_count,
                nut_count,
            )

        if bolt_count > nut_count:
            return "bolt", bolt_count

        return "nut", nut_count

    def _tracked_position_statistics(self):
        """
        Use the median center for robustness against one small localization
        outlier. Stability is still checked using the maximum distance.
        """
        if not self.stable_history:
            return None, None, None

        arr = np.vstack(
            self.stable_history
        )

        center = np.median(
            arr,
            axis=0,
        )

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
            center,
            max_dev,
            stable,
        )

    def _stable_detection_angle_deg(self):
        """
        Rectangle orientation is 180-degree periodic.
        Average it with a doubled-angle circular mean.
        """
        if not self.angle_history:
            return None

        radians = np.radians(
            np.asarray(
                self.angle_history,
                dtype=np.float64,
            )
        )

        c = float(
            np.mean(
                np.cos(2.0 * radians)
            )
        )
        s = float(
            np.mean(
                np.sin(2.0 * radians)
            )
        )

        return float(
            0.5
            * math.degrees(
                math.atan2(s, c)
            )
        )

    def _image_direction_to_base_yaw_deg(
        self,
        u,
        v,
        image_angle_deg,
    ):
        """
        Transform an image-plane direction through Homography and return the
        corresponding absolute direction angle in the robot Base XY plane.
        """
        if self.H is None:
            return None

        theta = math.radians(
            float(image_angle_deg)
        )

        length = self.angle_vector_length_px

        u2 = float(u) + length * math.cos(theta)
        v2 = float(v) + length * math.sin(theta)

        x1, y1 = self.pixel_to_base_xy(
            float(u),
            float(v),
        )
        x2, y2 = self.pixel_to_base_xy(
            u2,
            v2,
        )

        return float(
            math.degrees(
                math.atan2(
                    y2 - y1,
                    x2 - x1,
                )
            )
        )

    def _pixel_angle_to_base_delta_yaw_deg(
        self,
        u,
        v,
        image_angle_deg,
    ):
        """
        Convert the detected IMAGE angle into a RELATIVE yaw rotation in Base.

        fixed_q is defined as the desired gripper orientation when image angle
        == 0 deg. Therefore we must NOT multiply fixed_q by the absolute Base
        direction. Instead:

            delta_yaw =
                BaseDirection(image_angle)
                - BaseDirection(image_angle=0)

        This preserves the taught HOME orientation when angle == 0 and only
        rotates the gripper around Base Z by the target's relative in-plane
        rotation.

        Because minAreaRect orientation is 180-degree periodic, delta is
        normalized to [-90, 90).
        """
        target_base_yaw = self._image_direction_to_base_yaw_deg(
            u,
            v,
            image_angle_deg,
        )

        reference_base_yaw = self._image_direction_to_base_yaw_deg(
            u,
            v,
            0.0,
        )

        if (
            target_base_yaw is None
            or reference_base_yaw is None
        ):
            return None

        delta = (
            target_base_yaw
            - reference_base_yaw
        )

        while delta >= 90.0:
            delta -= 180.0

        while delta < -90.0:
            delta += 180.0

        return float(delta)

    def _finalize_cycle(
        self,
        msg,
        debug,
    ):
        """
        Finish exactly one camera burst.

        TARGET:
            enough observations belong to the SAME spatial track,
            class vote is consistent, and position is stable.

        NONE:
            no object-like contour was seen at all.

        INVALID:
            an object was seen, but association / class / stability was not
            reliable enough for robot motion.
        """
        matched_frames = len(
            self.stable_history
        )

        if matched_frames == 0:
            if self.cycle_had_object_like_contour:
                self._finish_positioning_cycle(
                    "invalid"
                )
            else:
                self._finish_positioning_cycle(
                    "none"
                )
            return

        if matched_frames < self.min_track_frames:
            self.get_logger().warning(
                "INVALID target track: "
                f"matched_frames={matched_frames} < "
                f"min_track_frames={self.min_track_frames}"
            )
            self._finish_positioning_cycle(
                "invalid"
            )
            return

        final_label, class_votes = (
            self._final_cycle_label()
        )

        if (
            final_label not in ("bolt", "nut")
            or class_votes
            < self.min_class_consistency_frames
        ):
            self.get_logger().warning(
                "INVALID class consensus: "
                f"history={self.class_history}, "
                f"winner={final_label}, "
                f"votes={class_votes}, "
                f"required={self.min_class_consistency_frames}"
            )
            self._finish_positioning_cycle(
                "invalid"
            )
            return

        center, max_dev, stable = (
            self._tracked_position_statistics()
        )

        if center is None:
            self._finish_positioning_cycle(
                "invalid"
            )
            return

        stable_u = float(
            center[0]
        )
        stable_v = float(
            center[1]
        )

        if (
            self.require_stable_position
            and not stable
        ):
            self.get_logger().warning(
                "INVALID position stability: "
                f"class={final_label}, "
                f"center=({stable_u:.2f},{stable_v:.2f}), "
                f"max_dev={max_dev:.2f}px > "
                f"{self.stable_pixel_tolerance:.2f}px"
            )
            self._finish_positioning_cycle(
                "invalid"
            )
            return

        if self.H is None:
            self.get_logger().warning(
                "INVALID: Homography is not loaded."
            )
            self._finish_positioning_cycle(
                "invalid"
            )
            return

        if self.target_z <= -10.0:
            self.get_logger().warning(
                "INVALID: target_z is not configured."
            )
            self._finish_positioning_cycle(
                "invalid"
            )
            return

        base_x, base_y = (
            self.pixel_to_base_xy(
                stable_u,
                stable_v,
            )
        )

        stable_angle_deg = (
            self._stable_detection_angle_deg()
        )

        q = self.fixed_q.copy()
        yaw_delta_deg = None

        use_angle_for_class = (
            self.use_detection_angle
            and (
                (
                    final_label == "bolt"
                    and self.use_detection_angle_for_bolt
                )
                or (
                    final_label == "nut"
                    and self.use_detection_angle_for_nut
                )
            )
        )

        if (
            use_angle_for_class
            and stable_angle_deg is not None
        ):
            yaw_delta_deg = (
                self._pixel_angle_to_base_delta_yaw_deg(
                    stable_u,
                    stable_v,
                    stable_angle_deg,
                )
            )

            if yaw_delta_deg is not None:
                yaw = math.radians(
                    yaw_delta_deg
                    + self.yaw_offset_deg
                )

                # Rotate in Base frame around Z while preserving the taught
                # vertical-down HOME orientation.
                q = quaternion_multiply(
                    yaw_quaternion(yaw),
                    q,
                )

                q = quaternion_normalize(
                    q
                )

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = (
            self.target_frame
        )

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

        final_class_msg = String()
        final_class_msg.data = final_label
        self.class_pub.publish(
            final_class_msg
        )

        angle_info = (
            "disabled"
            if not use_angle_for_class
            else (
                f"image={stable_angle_deg:.1f}deg, "
                f"base_delta={yaw_delta_deg:.1f}deg"
                if yaw_delta_deg is not None
                else "unavailable"
            )
        )

        self.get_logger().info(
            "FINAL TRACKED TARGET: "
            f"class={final_label} "
            f"votes={class_votes}/{matched_frames}, "
            f"pixel=({stable_u:.2f},{stable_v:.2f}), "
            f"max_dev={max_dev:.2f}px, "
            f"base=({base_x:.4f},{base_y:.4f})m, "
            f"angle={angle_info}"
        )

        self._finish_positioning_cycle(
            f"target:{final_label}"
        )

    # ================================================================
    # Main image callback
    # ================================================================
    def on_image(self, msg):
        try:
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
                    cv2.contourArea(
                        contour
                    )
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

                # Something object-like exists in this frame.
                self.cycle_had_object_like_contour = True

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

                if self.classifier is not None:
                    label = str(
                        class_name
                    ).strip().lower()
                else:
                    label = str(
                        self.labels.get(
                            class_id,
                            f"class_{class_id}",
                        )
                    ).strip().lower()

                if (
                    self.classifier is not None
                    and self.target_class_id >= 0
                    and class_id
                    != self.target_class_id
                ):
                    continue

                if (
                    self.classifier is None
                    and not self.publish_when_no_svm
                ):
                    continue

                # Only recognized sorting classes are eligible targets.
                if label not in (
                    "bolt",
                    "nut",
                ):
                    continue

                # Classification confidence now affects selection.
                if (
                    self.classifier is not None
                    and float(class_score)
                    < self.min_target_score
                ):
                    continue

                angle_deg = (
                    normalize_rect_angle(
                        rect
                    )
                )

                candidates.append(
                    {
                        "area": area,
                        "contour": contour,
                        "u": float(u),
                        "v": float(v),
                        "angle_deg": float(
                            angle_deg
                        ),
                        "label": label,
                        "score": float(
                            class_score
                        ),
                        "class_id": int(
                            class_id
                        ),
                    }
                )

            selected = (
                self._match_tracked_candidate(
                    candidates
                )
            )

            # Draw all eligible candidates in blue.
            for candidate in candidates:
                rect = cv2.minAreaRect(
                    candidate["contour"]
                )
                box = cv2.boxPoints(
                    rect
                ).astype(np.int32)

                cv2.drawContours(
                    debug,
                    [box],
                    0,
                    (255, 0, 0),
                    1,
                )

            if selected is not None:
                u = selected["u"]
                v = selected["v"]

                # Update the locked physical target.
                self.track_center = (
                    u,
                    v,
                )

                self.stable_history.append(
                    np.array(
                        [u, v],
                        dtype=np.float64,
                    )
                )
                self.angle_history.append(
                    selected["angle_deg"]
                )
                self.class_history.append(
                    selected["label"]
                )
                self.score_history.append(
                    selected["score"]
                )

                box = cv2.boxPoints(
                    cv2.minAreaRect(
                        selected["contour"]
                    )
                ).astype(np.int32)

                cv2.drawContours(
                    debug,
                    [box],
                    0,
                    (0, 255, 0),
                    3,
                )

                cv2.circle(
                    debug,
                    (
                        int(round(u)),
                        int(round(v)),
                    ),
                    6,
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    debug,
                    (
                        f"TRACK {selected['label']} "
                        f"score={selected['score']:.3f} "
                        f"u={u:.1f} v={v:.1f} "
                        f"a={selected['angle_deg']:.1f}"
                    ),
                    (
                        max(0, int(u) - 210),
                        max(25, int(v) - 20),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                class_msg = String()
                class_msg.data = (
                    selected["label"]
                )
                self.class_pub.publish(
                    class_msg
                )

                if self.H is not None:
                    base_x, base_y = (
                        self.pixel_to_base_xy(
                            u,
                            v,
                        )
                    )

                    self.get_logger().info(
                        "Tracked: "
                        f"class={selected['label']}, "
                        f"score={selected['score']:.3f}, "
                        f"pixel=({u:.1f},{v:.1f}), "
                        f"base=({base_x:.4f},{base_y:.4f})m, "
                        f"angle={selected['angle_deg']:.1f}deg",
                        throttle_duration_sec=1.0,
                    )
            else:
                # A track already exists but no candidate is close enough:
                # do NOT switch to another object.
                if self.track_center is not None:
                    self.get_logger().warning(
                        "No candidate matched the locked target "
                        f"within {self.target_match_radius_px:.1f}px "
                        f"on frame {self.cycle_frame_count}/"
                        f"{self.stable_frames}."
                    )

            cv2.putText(
                debug,
                (
                    f"cycle={self.cycle_frame_count}/"
                    f"{self.stable_frames} "
                    f"matched={len(self.stable_history)}"
                ),
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if (
                self.cycle_frame_count
                >= self.stable_frames
            ):
                self._finalize_cycle(
                    msg,
                    debug,
                )

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
