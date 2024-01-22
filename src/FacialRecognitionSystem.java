import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Rect;

import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import static org.opencv.highgui.HighGui.namedWindow;
import static org.opencv.highgui.HighGui.imshow;
import static org.opencv.highgui.HighGui.waitKey;

public class FacialRecognitionSystem {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the pre-trained face detection classifier (you may need to change the path)
        CascadeClassifier faceCascade = new CascadeClassifier("C:/lpu/java development/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml");

        // Open a video capture object (0 indicates the default camera)
        VideoCapture videoCapture = new VideoCapture(0);

        if (!videoCapture.isOpened()) {
            System.out.println("Error: Camera not detected!");
            return;
        }

        // Create an empty matrix to store frames
        Mat frame = new Mat();

        // Create a window to display the output
        namedWindow("Face Recognition");

        while (true) {
            // Capture a frame from the camera
            videoCapture.read(frame);

            // Convert the frame to grayscale for face detection
            Mat grayFrame = new Mat();
            Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

            // Detect faces in the frame
            MatOfRect faces = new MatOfRect();
            faceCascade.detectMultiScale(grayFrame, faces);

            // Draw rectangles around detected faces
            for (Rect rect : faces.toArray()) {
                Imgproc.rectangle(frame, new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0), 3);
            }

            // Display the frame
            imshow("Face Recognition", frame);

            // Break the loop if 'ESC' key is pressed
            if (waitKey(10) == 27) {
                break;
            }
        }

        // Release the video capture object
        videoCapture.release();
    }
}
