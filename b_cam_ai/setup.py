from setuptools import setup
import os
from glob import glob

package_name = 'b_cam_ai'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name], # Dòng này sẽ tự copy models.py, datasets.py nếu chúng nằm trong folder b_cam_ai
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # 1. Copy Weights
        (os.path.join('share', package_name, 'weights'), glob('weights/*.pth')),
        
        # 2. Copy Data (vn_templates.txt) vào đúng vị trí để Tokenizer tìm thấy
        # datasets.py tìm file theo đường dẫn tương đối, nên ta copy vào thư mục cài đặt python
        # Tuy nhiên, cách an toàn nhất trong ROS là copy vào share, 
        # nhưng để không phải sửa datasets.py, ta cần copy vào thư mục package python.
        # Setuptools 'package_data' xử lý việc này tốt hơn data_files cho file nội bộ.
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orin',
    maintainer_email='user@todo.todo',
    description='Referring Image Segmentation Node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'human_segment_node = b_cam_ai.human_segment_node:main',
            'referring_segmentation_node = b_cam_ai.referring_segmentation_node:main', # <-- THÊM DÒNG NÀY
        ],
    },
    # QUAN TRỌNG: Để copy vn_templates.txt vào cùng chỗ với code python
    package_data={
        'b_cam_ai': ['data/*.txt'], 
    },
)