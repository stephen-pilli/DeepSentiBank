Use under BSD Lisence.
Run python sentiBank.py, it will explain itself. CPU/GPU are both supported.
You can run the example image by:
python sentiBank.py test_image.jpg

The output should be a json file, containing 2,089 ranked concept scores, and a 4096-dimension feature (fc7).

Under windows 7+, you can probably run above command directly. Otherwise you may need to compile Caffe. 
Before compiling, put the extract_nfeatures.cpp under caffe/tools. After compiling, copy/link caffe/build/tools/extract_nfeatures.bin or exe to DeepSentiBank folder.
There is also a .m file that can read the raw feature file (fc7.dat prob.dat) generated from the executable extract_nfeatures. into matlab.

Please cite:
@article{chen2014deepsentibank,
  title={Deepsentibank: Visual sentiment concept classification with deep convolutional neural networks},
  author={Chen, Tao and Borth, Damian and Darrell, Trevor and Chang, Shih-Fu},
  journal={arXiv preprint arXiv:1410.8586},
  year={2014}
}

ubuntu 17.04 or greater {

sudo apt-get install libboost-dev
sudo apt-get install libcaffe-cpu-dev
sudo apt-get install libgflags-dev
sudo apt install libgoogle-glog-dev

}

nvcc extract_nfeatures.cpp -o extract_nfeatures -lprotobuf -lglog -lpthread -lboost_system -lcaffe

python sentiBank.py test_vso/image_path_list.txt GPU 0

GPU LIB REQ:
cuda 10.2
cublas
cudnn 7.5 for 10.2

caffe compilation requires latest version of cmake.

caffe installtion link
https://caffe.berkeleyvision.org/install_apt.html

Original Source Code and weight files are available at 
https://www.dropbox.com/sh/kzqkd7c94aarijd/AADzE0r19XpvzBw_K5Bbeq1Da?dl=0&file_subpath=%2FDeepSentiBank&preview=DeepSentiBank_works_with_Caffe_rc2.zip


