FROM centos:latest

RUN yum update -y
RUN yum install epel-release -y
RUN yum -y install gcc gcc-c++ 
RUN yum -y python3-pip python3-devel atlas atlas-devel gcc-gfortran openssl-devel libffi-devel

RUN yum install --upgrade tensorflow -y
RUN pip3 install numpy -y
RUN pip3 install keras -y
RUN pip3 install h5py opencv-python pillow scipy -y

