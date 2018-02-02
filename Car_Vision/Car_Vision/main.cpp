#include <dlib\opencv.h>
#include <opencv2\highgui\highgui.hpp>
#include <dlib\image_processing\frontal_face_detector.h>
#include <dlib\image_processing\render_face_detections.h>
#include <dlib\image_processing.h>
#include <dlib\gui_widgets.h>



using namespace cv;
using namespace dlib;
using namespace std;

int main(int argc, char** argv) {

	try {
		VideoCapture cap(0);
		if (!cap.isOpened()) {
			return -1;
		}

		image_window win;
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("S:\\dlib-19.9\\models\\shape_predictor_68_face_landmarks.dat") >> pose_model;

		while (!win.is_closed()) {
			Mat temp;
			if (!cap.read(temp)) {
				break;
			}

			cv_image<bgr_pixel> cimg(temp);
			std::vector<rectangle> face = detector(cimg);
			std::vector<full_object_detection> shapes;
			for (unsigned long i = 0; i < face.size(); ++i) {
				shapes.push_back(pose_model(cimg, face[i]));
			}

			win.clear_overlay();
			win.set_image(cimg);
			win.add_overlay(render_face_detections(shapes));
		}
	}
	catch (serialization_error& e) {
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e) {
		cout << e.what() << endl;
	}
	
}

