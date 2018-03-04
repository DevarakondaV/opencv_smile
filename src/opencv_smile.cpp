//Include files
//Standard
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <string>
//OpenCV
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"


//Namespaces
using namespace std;
using namespace cv;


//Function Declerations
Mat detectAndDisplay(Mat frame);
double Area(int w,int h);
int checksmile(Mat frame);
void print_inst();

//Some globals
Ptr<ml::ANN_MLP> nn = Algorithm::load<ml::ANN_MLP>("../res/nn1.yml");
cv::String face_cas = "../res/haarcascade_frontalface_default.xml";
CascadeClassifier face_cascade;

int main(int argc,char** argv)
{
	int opt;
	VideoCapture cap;
	VideoWriter v_writer;
	bool capture_vid = false;
	string file_name;
	while((opt = getopt(argc,argv,"w:f:s::h")) != -1)
	{	
		switch(opt)
		{
			case 'w':
				file_name = "webcam";
				cap = VideoCapture(atoi(optarg));
				break;
			case 'f':
				file_name = optarg;
				cap = VideoCapture(file_name);
				break;
			case 's':
				capture_vid = true; 
				break;
			case '?':
			case 'h':
			default:
				print_inst();
				return 0;
		}
	}
				
	
	//Frame contains video data
	Mat frame;

	
	if (!face_cascade.load(face_cas)){printf("--(1) Unable to find cascade classifier. Check /res for haarcascade_frontalface_default.xml\n"); return -1;};	

	if (cap.isOpened()) //check that webcam/video file was opened properly
	{
		if (capture_vid == true) {
			int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
			int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
			double fps = cap.get(CV_CAP_PROP_FPS);
			string output_file = file_name;
			int start_idx = output_file.find(".avi");
			if (start_idx != -1) {
				output_file.erase(start_idx,4);
			}
			output_file = output_file+"_copy.avi";
			v_writer = VideoWriter(output_file,CV_FOURCC('M','J','P','G'),fps,Size(frame_width,frame_height));
		}
		while(true)
		{
			cap.read(frame); //Read captured data into frame
			if (!frame.empty())
			{
				Mat guess_frame = detectAndDisplay(frame);
				if (capture_vid == true) {
					v_writer.write(guess_frame);
				}
		       	}
			else
			{
				printf("---(!) No captured frame/End of Video --Break!\n"); break;
			}
			int c = waitKey(10);
			if ((char) c == 'q') {break;} //stop execution
		}
	}
	return 0;
}

//Detects if there is a face in frame and predicts for smile.
Mat detectAndDisplay(Mat frame)
{
	
	std::vector<Rect> faces;
	Mat frameCopy = frame;	
	face_cascade.detectMultiScale(frame,faces,1.2,2,0,Size(50,50)); //Faces smaller than pixel size 50x50 will not be detected
	int x,y,h,w;

		
	size_t ind = 0;
	int smile; //Condition checked if there is a smile or not
	if (faces.size() > 0 && ind >=0 )
	{
		Mat cropped,grey,resized;
		x = faces[ind].x; y = faces[ind].y; h = y+faces[ind].height; w = x+faces[ind].width;
		if ( 0 <= x && 0 <= y && h <= frame.rows && w <= frame.cols && 0<= faces[ind].width && 0<= faces[ind].height)
		{
			cropped = frameCopy(Rect(x,y,faces[ind].width,faces[ind].height)); //Cropping face pixels from  frame
		}

		//Preprocessing before neural net
		cvtColor(cropped,grey, COLOR_BGR2GRAY);
		resize(grey,resized,Size(50,50));
		
		smile = checksmile(resized);
		Scalar col;
		if (smile == 1)
			col = Scalar(0,255,0);
		else
			col = Scalar(0,0,255); 
		rectangle(frame,Point (x,y),Point(w,h),col,2,8,0); //Adding boundary box.
	}
	imshow("window", frame);
	return frame;
}

double Area(int w,int h){
	return w*h;
}

//Uses Neural net to predict if there is a smile inside frame
int checksmile(Mat frame)
{
	clock_t start,end;
	double cpu_time_used;

	frame.convertTo(frame,CV_32F);
	Mat feature;
	frame = frame.reshape(1,1);
	normalize(frame,feature,0,1,NORM_MINMAX,-1);
	
	int rtnval =  nn->predict(feature);
	return rtnval;
}

void print_inst()
{
	cout << "SmileNN Help\n";
	cout << "-w \t Run with webcam. Requires devpath number. Example input: ./SmileNN -f [0]\n";
	cout << "-f \t Run with file. Requires filename. Example input: ./SmileNN -f [Filepath]\n";
	cout << "-s \t Save video file with prediction\n";
	cout << "-h \t help\n";
}

