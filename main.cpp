#include "opencv2/objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>

const int KEY_SPACE = 32;
const int KEY_ESC = 27;

using namespace std;
using namespace cv;

// FUNCTION PROTOTYPES ////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////
// Function to support user in the usage of the application
///////////////////////////////////////////////////////////////////////////////////////////////////
static void help();
///////////////////////////////////////////////////////////////////////////////////////////////////
// Function that pre-processes the input stream in order to make it possible to detect
// moving blobs of pixels. It takes the previous frame and the current and returns a resulting
// comparison image
///////////////////////////////////////////////////////////////////////////////////////////////////
Mat prepareFrameForDetection(Mat *currentFrame, Mat *previousFrame);
///////////////////////////////////////////////////////////////////////////////////////////////////
// Function that returns the number of detected objects within a Region Of Interest as defined by
// the cascade classifier
///////////////////////////////////////////////////////////////////////////////////////////////////
int detectAndDraw( Mat& frame, Mat& img, CascadeClassifier& cascade, Point ROIOrigin, Size size);
///////////////////////////////////////////////////////////////////////////////////////////////////
// Function to watermark the count of detected vehicles on the display
///////////////////////////////////////////////////////////////////////////////////////////////////
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy);
///////////////////////////////////////////////////////////////////////////////////////////////////
// Function to output to an image file (.png) the region created by the detection
///////////////////////////////////////////////////////////////////////////////////////////////////
int outputPositiveSample(int n, Mat ROI);
///////////////////////////////////////////////////////////////////////////////////////////////////
// Function to obtain the current date and time for logging
///////////////////////////////////////////////////////////////////////////////////////////////////
string getCurrentDateTime(string s);
///////////////////////////////////////////////////////////////////////////////////////////////////
// Function to log to a file the date and count when detection
///////////////////////////////////////////////////////////////////////////////////////////////////
void Logger( string logMsg );

string cascadeName;
string inputName;
string trainMode;
Size FramesSize(320,240);
int vehicleDetectionsThreshold = 10;
int emptyDetectionThreshold = 9;
int SpanThreshold = 15;

int main( int argc, const char** argv )
{
    //Captures the video input
    VideoCapture capture;
    //OpenCV's image containers
    Mat previousFrame, currentFrame, image;
    //Use the CascadeClassifier class to detect objects in a video stream. Particularly, we will use the functions:
    //load to load a .xml classifier file. It can be either a Haar or a LBP classifer
    //detectMultiScale to perform the detection.
    CascadeClassifier cascade;

    //Number of counted vehicles in the video sequence
    int totalVehicleCount = 0;
    //Number of detected vehicles detected in a frame
    int detectedVehicleCount = 0;
    //Number of frames without vehicles detected
    int emptyFrameCount = 0;
    //Number of frames since the first frame with vechicles was detected
    int frameSpanCount = 0;
    //Toggled when a number of vehicles has been detected recently
    bool trackingCar = false;
    //Absolute position in the frame sequence
    int frameNum = 0;
    //The CommandLineParser class is designed for command line arguments parsing
    CommandLineParser parser(argc, argv,
        "{help h||}"
        "{cascade|../../data/cars.xml|}"
        "{@filename||}"
        "{train||}"
    );
    //Check if the user required help
    if (parser.has("help"))
    {
        help();
        return 0;
    }

    //Get the cascade and the video stream input
    cascadeName = parser.get<string>("cascade");
    inputName = parser.get<string>("@filename");

    trainMode = parser.get<string>("train");

    //Return any errors found
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    //Check that the cascade was loaded
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }

    //Check that the user didn't input a video stream. In this case we will try to use the web cam
    if( inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1) )
    {
        int camera = inputName.empty() ? 0 : inputName[0] - '0';
        if(!capture.open(camera))
            cout << "Capture from camera #" <<  camera << " didn't work" << endl;
    }
    //If there is an input file, we try to use it
    else if( inputName.size() )
    {
        image = imread( inputName, 1 );
        if( image.empty() )
        {
            if(!capture.open( inputName ))
                cout << "Could not read " << inputName << endl;
        }
    }

    //****************************************************************
    //If succeeded to find a video stream, we can get started!!
    //****************************************************************
    if( capture.isOpened() )
     {
        cout << "Video capturing has been started ..." << endl;
        //Get the very first video frame
        capture >> previousFrame;
        //Resize the frame to fit the resolution of the trained cascade classifier
        resize(previousFrame,previousFrame,FramesSize);

        //Start analyzing the video capture
        for(;;)
        {
            frameNum++;
            //Report to console for debugging purposes
            cout << "**************New frame ***************" << endl;
            //Timer for measuring detection speed
            double t = 0;
            t = (double)getTickCount();

            //Parse the current video capture and its next frame into a pair of Mat elements
            capture >> currentFrame;
            //Verify that we are not at the end of the sequence
            if( currentFrame.empty() ){
              break;
            }
            //Resize the current frame to fit the resolution of the cascade classifier
            resize(currentFrame,currentFrame,FramesSize);
            //Create a copy of the current frame for further visualization and presentation
             Mat currentFrameDisplay = currentFrame.clone();
            //Compare this frame with the previous one to get a single image with the moving pixels
            Mat imgThresh = prepareFrameForDetection(&currentFrame, &previousFrame);
            //Create an array of contours
            vector<vector<Point>> currentContours;
            /*The function findContours retrieves contours from the binary
            image using the algorithm by Satoshi Suzuki and others. */
            findContours(imgThresh, currentContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            //Create an array of convex hulls with the size of the contour
            vector<vector<Point>> convexHulls(currentContours.size());
            //Iterate  through the array of contours and create a convex hull
            //out of each one of them.
            for (unsigned int i = 0; i < currentContours.size(); i++)
            {
              /*The function cv::convexHull finds the convex hull of a 2D point set
              using the Sklansky's algorithm that has O(N logN) complexity
              in the current implementation. */
              convexHull(currentContours[i], convexHulls[i]);
            }
            //Iterate through the array of convexhulls found
            for (auto &convexHull : convexHulls)
            {
              //Create a region of interest from the convex hull's bounding box
              //in order to analyze whether it contains vehicles in it
              Rect blobBoundingBox = boundingRect(convexHull);

              //We are only interested in applying detection to those blobs whose dimensions are significant.
              //Size ratios too small normally indicate perturbations in the illumination and other artifacts
              double significantBlobRatioMin = 0.2;
              double significantBlobRatioMax = 0.8;

              if ((blobBoundingBox.area() > FramesSize.area() * significantBlobRatioMin) && (blobBoundingBox.area() < FramesSize.area() * significantBlobRatioMax)){
                //The region of interest must be an OpenCV matrix
                Mat ROI;
                //Create a dense array to hold the grayed version of the frame that we will pass onto the cascade classifier
                Mat grayFrame;
                //Populate the matrix with the gray scale information of the current frame
                cvtColor(currentFrame, grayFrame, COLOR_BGR2GRAY);
                //We will fill it in with the gray scale contents within the rectangle
                ROI = grayFrame(blobBoundingBox);
                //Count the number of vehicles in the blob by means of a cascade
                int blobVehicleCount = detectAndDraw(currentFrame, ROI, cascade, Point(blobBoundingBox.x, blobBoundingBox.y), FramesSize);
                //If there are any vehicles, the blob is interesting
                if (blobVehicleCount > 0)
                {
                  cout << "It's a car!!!" << endl;
                  detectedVehicleCount++;
                  emptyFrameCount = 0;
                  trackingCar =true;
                  //Display a rectangle around the pixel blob
                  rectangle(currentFrameDisplay, blobBoundingBox, Scalar(0, 0, 255), 8, 8, 0);
                  if (!trainMode.empty()){
                    //Create an image out of this blob for further training of the cascade
                    outputPositiveSample(frameNum,ROI);
                  }
                }
                //Picture the blob on the frame for visualization
                rectangle(currentFrameDisplay, blobBoundingBox, Scalar(255, 255, 0), 2, 8, 0);
              }

            }
            //Check whether we are tracking a car so we can increase the number of valuable frames
            if (trackingCar){
              frameSpanCount ++;
            }
            //Cars are only accounted for after a safety distance is taken. Besides, there has to be a minimum of positive
            //detections within a given time span
            if (detectedVehicleCount > vehicleDetectionsThreshold &&
                emptyFrameCount > emptyDetectionThreshold &&
                frameSpanCount > SpanThreshold){
              //Increase the count of vehicles in the ramp
              totalVehicleCount ++ ;
              //Reset the negative detections and the positive
              emptyFrameCount = 0;
              detectedVehicleCount = 0;
              //Output the count to a file
              Logger(to_string(totalVehicleCount));
            }
            //In order to know whether we are detecting a car or something else, enough positive detections have to happen in a row
            if (emptyFrameCount > vehicleDetectionsThreshold){
              detectedVehicleCount = 0;
              trackingCar = false;
              frameSpanCount = 0;
            }
            //Counter for resetting the tracker
            emptyFrameCount++;
            //Count the time passed
            t = (double)getTickCount() - t;
            //Report results to console
            printf( "detection time = %g ms\n", t*1000/getTickFrequency());
            cout << "Current detection number = " << detectedVehicleCount << endl;
            cout << "Counted vehicles = " << totalVehicleCount << endl;
            cout << "Frames without vehicles = " << emptyFrameCount << endl;
            cout << "Frames span = " << frameSpanCount << endl;
            cout << "Tracking = " << trackingCar << endl;
            //Visualize the results
            drawCarCountOnImage(totalVehicleCount, currentFrameDisplay);
            imshow("Capture", currentFrameDisplay);

            // Prepare for the next iteration
            swap(previousFrame,currentFrame);

            //Get some user input in case it is needed to exit or freeze
            char c = (char)waitKey(33);

            if( c == 27 || c == 'q' || c == 'Q' )
                 break;
            if( c == KEY_SPACE)
                 waitKey(0);
        }
    }
    return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
static void help()
{
    cout << "\nThis program employs OpenCV's cascade recognizer. Now you can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "Usage:\n"
            "./OCVehicleSpooring [--cascade=<cascade_path> this is the primary trained classifier]\n"
               "   [--train]\n"
               "   [filename|camera_index]\n\n"
            "./OCVehicleSpooring --cascade=\"../../data/cascade.xml\" \n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Mat prepareFrameForDetection(Mat *currentFrame, Mat *previousFrame)
{
  //Create a copy of the frames for further edition and segmentation of the blobs
  Mat previousFrameForeGround = previousFrame->clone();
  Mat currentFrameForeGround = currentFrame->clone();
  //Create a dense array for comparing one frame to the next
  Mat imgDifference;
  //Create a dense array to perform morphological operations on it (Erosion and Dilation)
  Mat imgThresh;

  //Remove the color from all copies of the frames
  cvtColor(previousFrameForeGround, previousFrameForeGround, CV_BGR2GRAY);
  cvtColor(currentFrameForeGround, currentFrameForeGround, CV_BGR2GRAY);
  //Eliminate edges by means of a gaussian blur
  GaussianBlur(previousFrameForeGround, previousFrameForeGround, Size(int(FramesSize.width * 0.015)+1, int(FramesSize.width * 0.015)+1), 0);
  GaussianBlur(currentFrameForeGround, currentFrameForeGround, Size(int(FramesSize.width * 0.015)+1, int(FramesSize.width * 0.015)+1), 0);
  //Store the difference into imgDifference
  absdiff(previousFrameForeGround, currentFrameForeGround, imgDifference);

  //Store the threshold function in imgThresh
  /*The function applies fixed-level thresholding to a single-channel array.
  The threshold function is typically used to get a bi-level (binary) image out of a grayscale image
  (cv::compare could be also used for this purpose) or for removing a noise,
  that is, filtering out pixels with too small or too large values.
  There are several types of thresholding supported by the function.
  They are determined by type parameter. */
  threshold(imgDifference,    //src – input array (single-channel, 8-bit or 32-bit floating point).
            imgThresh,        //dst – output array of the same size and type as src.
            15,               //thresh – threshold value.
            255.0,            //maxval – maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
            CV_THRESH_BINARY  //type – thresholding type (see documentation).
            );

  //Create a structuring rectangle shape to use in the dilation and erosion operations
  Mat structuringElement = getStructuringElement(MORPH_RECT, Size(int(FramesSize.width * 0.05), int(FramesSize.height * 0.05)));
  /*The most basic morphological operations are two: Erosion and Dilation. They have a wide array of uses, i.e. :
    -Removing noise
    -Isolation of individual elements and joining disparate elements in an image.
    -Finding of intensity bumps or holes in an image.
    Here we will use them repeatedly to sharpen up the possible regions of interest over the 10x10 rectangle  */
  for (unsigned int i = 0; i < 2; i++)
  {
      dilate(imgThresh, imgThresh, structuringElement);
      dilate(imgThresh, imgThresh, structuringElement);
      erode(imgThresh, imgThresh, structuringElement);
  }

  return imgThresh;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
int detectAndDraw( Mat& frame, Mat& img, CascadeClassifier& cascade, Point ROIOrigin, Size size)
{

    vector<Rect> vehicles;
    //Utilize OpenCV's Cascade classifer
    cascade.detectMultiScale(img,     // image	Matrix of the type CV_8U containing an image where objects are detected.
                            vehicles, // objects	Vector of rectangles where each rectangle contains the detected object, the rectangles may be partially outside the original image.
                            1.1,      // scaleFactor	Parameter specifying how much the image size is reduced at each image scale.
                            4,        // minNeighbors	Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        //|CASCADE_DO_CANNY_PRUNING,
        CASCADE_FIND_BIGGEST_OBJECT,
        //|CASCADE_DO_ROUGH_SEARCH,
        //|CASCADE_SCALE_IMAGE,       // flags	Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
        Size(int(size.width * 0.25), int(size.height * 0.25)),  // minSize	Minimum possible object size. Objects smaller than that are ignored.
        size                         // maxSize	Maximum possible object size. Objects larger than that are ignored. If maxSize == minSize model is evaluated on single scale.
      );

    return vehicles.size();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawCarCountOnImage(int &count, Mat &imgFrame) {
    //Define the text parameters
    int fontFace = CV_FONT_HERSHEY_SIMPLEX;
    double fontScale = (imgFrame.rows * imgFrame.cols) / 100000.0;
    int fontThickness = (int)::round(fontScale * 5);
    Size textSize = getTextSize(to_string(count), fontFace, fontScale, fontThickness, 0);
    Point textBottomLeftPosition;
    //Locate the text in the frame
    textBottomLeftPosition.x = imgFrame.cols - 1 - (int)((double)textSize.width * 1.25);
    textBottomLeftPosition.y = (int)::round(imgFrame.rows * 0.9);
    //The OpenCV function putText renders the specified text string in the image.
    putText(imgFrame,                 //img – Image.
            to_string(count),         //text – Text string to be drawn.
            textBottomLeftPosition,   //org – Bottom-left corner of the text string in the image.
            fontFace,                 //fontFace – Font type. One of FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX, FONT_HERSHEY_COMPLEX, FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL, FONT_HERSHEY_SCRIPT_SIMPLEX, or FONT_HERSHEY_SCRIPT_COMPLEX, where each of the font ID’s can be combined with FONT_ITALIC to get the slanted letters.
            fontScale,                //fontScale – Font scale factor that is multiplied by the font-specific base size.
            Scalar(0.0, 200.0, 0.0),  //color – Text color.
            fontThickness,            //thickness – Thickness of the lines used to draw a text.
            false                     //bottomLeftOrigin – When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.
          );

}

///////////////////////////////////////////////////////////////////////////////////////////////////
int outputPositiveSample(int n, Mat ROI)
{
    //Declare the parameters for OpenCV's imwrite function
    stringstream filename;
    vector<int> compression_params;

    //The images will be stored in the same folder as the executable
    filename << "pos-" << n << ".png";
    //We will make them .png. For PNG, it can be the compression level ( CV_IMWRITE_PNG_COMPRESSION ) from 0 to 9.
    //A higher value means a smaller size and longer compression time. Default value is 3. We set it to 9 for efficiency
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    try {
      //The function imwrite saves the image to the specified file.
      //The image format is chosen based on the filename extension.
      //Only 8-bit (or 16-bit unsigned (CV_16U) in case of PNG, JPEG 2000, and TIFF)
      //single-channel or 3-channel (with ‘BGR’ channel order) images can be saved using this function.
      imwrite(filename.str(), ROI, compression_params);
    }
    catch (runtime_error& ex) {
      fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
      return 1;
    }

    return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////
string getCurrentDateTime(string s)
{
    time_t now = time(0);
    struct tm  tstruct;
    char  buf[80];
    tstruct = *localtime(&now);
    if(s=="now")
        strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);
    else if(s=="date")
        strftime(buf, sizeof(buf), "%Y-%m-%d", &tstruct);
    return string(buf);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Logger(string logMsg)
{

    string filePath = "log_"+getCurrentDateTime("date")+".txt";
    string now = getCurrentDateTime("now");
    ofstream ofs(filePath.c_str(), std::ios_base::out | std::ios_base::app );
    ofs << now << '\t' << logMsg << '\n';
    ofs.close();
}
