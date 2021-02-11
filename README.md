# 🔬 OpenCV를 이용한 지문 인식 프로젝트 
Fingerprint detection using OpenCV

# 프로젝트 개요
 * OpenCV를 이용한 지문 영상 취득과 다양한 영상 처리 알고리즘들을 활용하여 이미지를 복원하고 특징점을 찾아 육안으로 특징점을 Counting한 결과와 비교 분석하여 성능을 검증하였습니다.
 * 원본 영상의 Orientation map를 취득하고 Gabor filter(사인 함수로 모듈레이션 된 Gaussian Filter)를 이용하여 끉어진 지문을 복원하는 역할을 맡음.
### 지문에는 다음과 같은 특징점들이 존재한다.
<img src = "http://3.bp.blogspot.com/-BEecRFn9fRM/T531N9a83SI/AAAAAAAAAKk/d0GHT8l6d7U/s1600/Thumb%2Bprint.png" width="400"/>


## Project Info ❓ :

 * IDE 　　　　　 : 	**Visual Studio 2019** - 2020-03
 * Language　　　: **C++**
 * SDK, Library　 　:	**OpenCV**

## Features ❗ :
 * 지문 프로젝트 : 이미지 강화 (정규화, Gabor 필터 적용)
 ``` c++
 // gabor filter parameter 
	double sig = 9, lm = 7.2, gm = 0.02, ps = 0;
	double theta;
	float ffi;

	/////Gabor filtering
	for (int m = special; m < temp.rows - special; m++) {
		for (int n = special; n < temp.cols - special; n++) {
			theta = stemp.at<float>(m, n);
			kernel3 = getGaborKernel(Size(kernel_size, kernel_size), sig, theta, lm, gm, ps, CV_32F);
			ffi = 0;
			for (int k = 0; k < kernel_size; k++) {
				for (int l = 0; l < kernel_size; l++) {
					ffi += temp.at<float>(m - special + k, n - special + l)*kernel3.at<float>(kernel_size - 1 - k, kernel_size - 1 - l);
				}
			}
			forgab.at<float>(m, n) = ffi / (kernel_size * kernel_size);
		}
	}
  ```


## Image ❗ :
### 이미지 처리 과정 단계
<img src="./Assets/image/image1.png" width="600"/>

### 처리 과정 단계 Part별 Result 
<img src="./Assets/image/image2.png" width="600"/>

### Gabor Filter & Thinning
<img src="./Assets/image/image4.png" width="600"/> 
<img src="./Assets/image/image5.png" width="600"/>

### 미팅 사진 👩‍👩‍👧‍👦
<img src="./Assets/image/image3.png" width="600"/>
