#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <vector>
#include <iostream>

constexpr int BOARD_WIDTH = 10;
constexpr int BOARD_HEIGHT = 7;
constexpr float BOARD_CELL_SIZE = 0.025f;

int main(int argc, char** argv)
{
	std::vector<cv::Mat> images;
	cv::Mat buffer;
	cv::VideoCapture camVideoCapture("./chessboard-720.mp4");
	cv::Matx33d kValues(988.0130764328934, 0, 420.9784087272643, 0, 1001.272688293928, 204.1115496365296, 0, 0, 1);
	std::vector<double> distanceCoefficient = { 0.2319312814294019, -4.133235939041602, 0.00254585949211152, 0.007092331493671674, 21.51409719800601 };

	int key = 0;
	bool ret = 0;
	cv::Mat buffer1, buffer2;

	std::vector<cv::Point3d>  vertex = { cv::Point3d(4 * BOARD_CELL_SIZE, 2 * BOARD_CELL_SIZE,  -2 * BOARD_CELL_SIZE) };
	std::vector<cv::Point3d> triangle_lower = { cv::Point3d(3 * BOARD_CELL_SIZE, 1 * BOARD_CELL_SIZE, 0),  cv::Point3d(5 * BOARD_CELL_SIZE, 1 * BOARD_CELL_SIZE, 0),
	cv::Point3d(4 * BOARD_CELL_SIZE, 3 * BOARD_CELL_SIZE, 0) };

	if (camVideoCapture.isOpened() != true)
	{
		__debugbreak();
	}

	std::vector<cv::Point3f> _3dPoints;

	for (size_t i = 0; i < BOARD_HEIGHT; i++)
	{
		for (size_t j = 0; j < BOARD_WIDTH; j++)
		{
			_3dPoints.push_back({ BOARD_CELL_SIZE * j, BOARD_CELL_SIZE * i, 0 });
		}
	}

	std::vector<cv::Point2d> pointBuffer;

	while (camVideoCapture.isOpened() == true)
	{
		ret = camVideoCapture.read(buffer);

		if (ret == false)
		{
			break;
		}
		else
		{
			ret = cv::findChessboardCorners(buffer, { BOARD_WIDTH, BOARD_HEIGHT }, pointBuffer, cv::CALIB_CB_ADAPTIVE_THRESH
				+ cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

			if (ret == true)
			{
				cv::Mat rvec, tvec;
				cv::solvePnP(_3dPoints, pointBuffer, kValues, distanceCoefficient, rvec, tvec);

				cv::Mat lowerLine, vertexBuffer;
				cv::projectPoints(triangle_lower, rvec, tvec, kValues, distanceCoefficient, lowerLine);
				cv::projectPoints(vertex, rvec, tvec, kValues, distanceCoefficient, vertexBuffer);
				lowerLine.reshape(1).convertTo(lowerLine, CV_32S);
				vertexBuffer.reshape(1).convertTo(vertexBuffer, CV_32S);

				cv::polylines(buffer, lowerLine, true, cv::Vec3b(255, 0, 0), 2);

				for (int i = 0; i < lowerLine.rows; i++)
				{
					cv::line(buffer, cv::Point(lowerLine.row(i)), cv::Point(vertexBuffer.row(0)), cv::Vec3b(0, 255, 0), 2);
				}

				cv::Mat _3DRotationMatrix;
				cv::Rodrigues(rvec, _3DRotationMatrix);

				double sy = sqrt(_3DRotationMatrix.at<double>(0, 0) * _3DRotationMatrix.at<double>(0, 0) +
						  _3DRotationMatrix.at<double>(1, 0) * _3DRotationMatrix.at<double>(1, 0));
				bool singular = sy < 1e-6;

				cv::Mat temp = -_3DRotationMatrix.t() * tvec;
				std::cout << "X, Y, Z: " << cv::Point3d(temp) << std::endl;

				if (singular != true) {
					std::cout << "roll: " << std::atan2(_3DRotationMatrix.at<double>(2, 1), _3DRotationMatrix.at<double>(2, 2)) << std::endl;
					std::cout << "pitch: " << std::atan2(-_3DRotationMatrix.at<double>(2, 0), sy) << std::endl;
					std::cout << "yaw: " << std::atan2(_3DRotationMatrix.at<double>(1, 0), _3DRotationMatrix.at<double>(0, 0)) << std::endl;
				}
				else {
					std::cout << "roll: " << std::atan2(-_3DRotationMatrix.at<double>(1, 2), _3DRotationMatrix.at<double>(1, 1)) << std::endl;
					std::cout << "pitch: " << std::atan2(-_3DRotationMatrix.at<double>(2, 0), sy) << std::endl;
					std::cout << "yaw: " << 0 << std::endl;
				}
			}

			cv::imshow("record", buffer);
		}

		cv::waitKey();
	}

	cv::destroyAllWindows();
	camVideoCapture.release();

	return 0;
}
