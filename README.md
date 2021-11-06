# libface-sface_detect-recognition-opencv
使用OpenCV部署libface人脸检测和SFace人脸识别，包含C++和Python两种版本的程序，仅仅只依赖OpenCV库就能运行

看到最近发布了opencv4.5.4里新增了人脸检测和人脸识别模块，这两个模块是由人脸识别领域的两位大牛设计的，
其中人脸检测是南科大的于仕琪老师设计的，人脸识别模块是北邮的邓伟洪教授设计的。
人脸检测示例程序在opencv-master/samples/dnn/face_detect.cpp里，起初我在win10系统里，在visual stdio 2019
里新建一个空项目，然后把opencv-master/samples/dnn/face_detect.cpp拷贝进来作为主程序，尝试编译，发现编译不通过。
仔细看代码可以发现face_detect.cpp里使用了类的继承和虚函数重写，这说明依赖包含了其他的.cpp和.hpp头文件的。因此我就编写一套程序，
人脸检测和人脸识别程序从opencv源码里剥离出来，只需编写一个main.cpp文件，就能运行人脸检测和人脸识别程序。整个程序只依赖opencv
就能运行，从而彻底摆脱对任何深度学习框架的依赖。
于仕琪老师设计的libface人脸检测，有一个特点就是输入图像的尺寸是动态的，也就是说对输入图像不需要做resize到固定尺寸，就能输入到
神经网络做推理的，此前我发布的一些人脸检测程序都没有做到这一点，而且模型文件.onnx只有336KB。因此，这套人脸检测模型是
非常有应用价值的。


模型文件在百度云盘，链接：https://pan.baidu.com/s/1NpoWW6UVEnmghxjxq4jCJw 
提取码：k7vz
