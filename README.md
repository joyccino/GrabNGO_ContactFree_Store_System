# 🌎 개발 환경 (Environment)
* OS: Ubuntu 20.04
* DB: Oracle
* 작업 Tool (Editor Tool): VSCode
* 프로그래밍 언어: Python
* 웹 표준 기술 : Flask
* 기타 사용 프레임 워크 : bootstrap, numpy, torch,torchvision, opencv, darknet, cx-oracle, pyqt, caffe, numba, dlib, face recognition, Flask, Flask-SQLAlchemy, paramico, pillow, tensorflow, SQLAlchemy, werkzeug, scikit-learn,bcrypt, config, cryptography, h5py, absl-py, keras, matplotlib,easydict,decorator, google-auth,google-pasta, ipython,jsonpatch, jsonpointer, keras-preprocessing, markdown,mtcnn,mxnet-mkl,pickleshare(pickle),cudnn, promt-toolkit, protobuf,scikit-image, scipy, tensorflow-gpu

# 코어 알고리즘 (Core Algorithm)

![IMG_1421 (1)-min](https://user-images.githubusercontent.com/67300266/103401695-411fb100-4b8d-11eb-818c-cc60bba187b6.gif) <br>

* <b>고객 인식 (Customer Recognition) </b> <br>
물체 판별기인 Yolov5와 인체 포즈 추정기인 OpenPose 연동하여 고객당 아이디 부여, 손의 위치 (파란색 점)을 추정하여 해당 아이디에 매핑.

* <b>상품 인식 (Product Detection) </b> <br>
1) 판매 상품에 대해 커스텀 학습시킨 모델을 Yolov5 에 적용하여 실시간 상품 인식 후에 상품당 Bounding Box 그려준 후 상품의 아이디 표시. <br>
2) 고객이 상품을 잡았을 경우 Bounding Box 지워주고 해당 고객의 버추얼 장바구니에 넣어줌.

* <b>상품을 가상 장바구니에서 제외하는 경우 (Product removal from a virtual cart)</b> <br>
1) 상품이 다시 판매대에 놓였을 경우 다시  Bounding Box 와 아이디 표시 <br>
2) 고객 가상 장바구니에서 제외 
