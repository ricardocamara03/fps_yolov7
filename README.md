This project contains the code used in the paper: Real-time Object Detection Performance Analysis Using YOLOv7 on Edge Devices.

Before running the code in this project, it is necessary to install the YOLOv7 prerequisites. To do this, follow the steps described in the official YOLOv7 repository: https://github.com/WongKinYiu/yolov7

Description of this project:
- models: Directory containing the model.
- utils: Required files are in this directory.
- iv_VideoCapturepy: Script containing the class used to capture images, which can perform synchronous and asynchronous capture.
- yolov7_detector.py: Script containing the class used to detect objects using YOLOv7.
- main.py: Main script that runs the application.

To run the project: $ python3 main.py output_file_name

------------------------------------------------------

The hardware utilization of NVIDIA devices is measured using the tegrastats tool.

Official documentation: https://docs.nvidia.com/drive/drive_os_5.1.6.1L/nvvib_docs/index.html#page/DRIVE_OS_Linux_SDK_Development_Guide/Utilities/util_tegrastats.html

In this project, the following command was used for measurement: sudo tegrastats --interval 20 --logfile ./output_file.txt

------------------------------------------------------

The power mode configuration change is done using the nvpmodel tool. Official documentation: https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#nvpmodel-gui

Example command to set energy mode 0: $ sudo /usr/sbin/nvpmodel -m 7
Commando to check the energy mode: $ sudo /usr/sbin/nvpmodel -q

------------------------------------------------------

If you have any doubt contact me by the following e-mail: ricardocamara03@gmail.com
