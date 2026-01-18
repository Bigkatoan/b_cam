from setuptools import setup
import os
from glob import glob # <--- 1. THÊM DÒNG NÀY

package_name = 'b_cam_ai'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # <--- 2. THÊM ĐOẠN NÀY ĐỂ COPY WEIGHTS ---
        # Copy tất cả file .pth trong thư mục weights vào share/b_cam_ai/weights
        (os.path.join('share', package_name, 'weights'), glob('weights/*.pth')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orin',
    maintainer_email='user@todo.todo',
    description='AI Segmentation Node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'human_segment_node = b_cam_ai.human_segment_node:main',
        ],
    },
)