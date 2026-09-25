from setuptools import find_packages, setup

package_name = 'autonomous_search_husky'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='labra',
    maintainer_email='labra@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "py_node_cuco = my_py_pkg.my_first_node:main",
            "mqtt_client = autonomous_search_husky.mqtt_client:main",
            "mqtt_receiver = autonomous_search_husky.mqtt_receiver:main",
            "yolo_detector = autonomous_search_husky.yolo_detector:main",
            "victim_localizer = autonomous_search_husky.victim_localizer:main",
            "mqtt_bridge = autonomous_search_husky.mqtt_bridge:main",
            "mission_manager = autonomous_search_husky.mission_manager:main"
        ],
    },
)
