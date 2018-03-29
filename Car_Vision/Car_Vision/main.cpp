#include <dlib\gui_widgets.h>
#include <dlib\image_processing.h>
#include <dlib\opencv.h>
#include <opencv\cv.hpp>
#include <iostream>
#include <fstream>



using namespace dlib;
using namespace cv;
using namespace std;

int main() {
	try {
		image_window win;
		Mat frame;
		VideoCapture capture("38335774.mp4");
		if (!capture.isOpened()) {
			return -1;
		}

		typedef scan_fhog_pyramid<pyramid_down<2>> image_scanner_type;
		object_detector<image_scanner_type> detector;
		deserialize("S:\\VS_Projects\\C-Project\\Car_Vision\\x64\\Release\\car_detector.svm") >> detector;

		while (!win.is_closed()) {
			capture >> frame;
			resize(frame, frame, Size(960, 540));

			cv_image<bgr_pixel> cimag(frame);
			std::vector<dlib::rectangle> rects = detector(cimag);

			cout << "Num Detections: " << rects.size() << endl;

			win.clear_overlay();
			win.set_image(cimag);
			win.add_overlay(rects, rgb_pixel(255, 0, 0));

			cout << "Hit enter to see the next image.";
			cin.get();
		}
	}

	catch (exception& e) {
		cout << "\n exception thrown!" << endl;
		cout << e.what() << endl;
	}

	return EXIT_SUCCESS;
}