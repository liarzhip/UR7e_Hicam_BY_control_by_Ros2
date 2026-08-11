from glob import glob
import os
from setuptools import find_packages, setup

package_name = "ur7e_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        # Runtime configuration files
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),

        # Trained SVM model bundle
        # Keep the complete model set together under models/:
        #   svm_model.xml
        #   svm_scaler.npz
        #   labels.json
        #   class_reference.npz
        #   model_meta.json
        (os.path.join("share", package_name, "models"), glob("models/*")),

        # Calibration files
        (os.path.join("share", package_name, "calibration"), glob("calibration/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="liar",
    maintainer_email="liar@example.com",
    description="HIKROBOT vision nodes for UR7e visual manipulation",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "hik_camera_node = ur7e_vision.hik_camera_node:main",
            "svm_detector_node = ur7e_vision.svm_detector_node:main",
        ],
    },
)
