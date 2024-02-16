#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

Mat convertToGrayScale(const Mat& inputImage)
{
    // Create an empty grayscale image
    Mat grayImage(inputImage.size(), CV_8UC1, Scalar(0));

    // Loop through each pixel and calculate the average of color channels
    for (int i = 0; i < inputImage.rows; ++i) 
    {
        for (int j = 0; j < inputImage.cols; ++j)
	{
            Vec3b color = inputImage.at<Vec3b>(i, j);

            // Calculate average intensity (grayscale value)
	    int grayValue;
	    if(inputImage.channels() == 1)
	    {
		grayValue = static_cast<int>(inputImage.at<int>(i,j)) / 3.0;
	    }
            else if(inputImage.channels() == 3)
	    {
		grayValue = static_cast<int>((color[0] + color[1] + color[2]) / 3.0);
	    }

            // Set the grayscale value in the output image
            grayImage.at<uchar>(i, j) = grayValue;
        }
    }

    return grayImage;
}

Mat applyGaussianBlur(const Mat& inputImage, const int& p_kernelSize, const double& sigma) 
{
    Mat blurredImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8U);

    int rows = inputImage.rows;
    int cols = inputImage.cols;
    // Calculate kernel size
    int kernelSize = p_kernelSize * sigma;

    // Create Gaussian kernel
    cv::Mat kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
    float sum = 0.0;

    for (int x = 0; x < kernelSize; ++x) 
    {
        for (int y = 0; y < kernelSize; ++y) 
	{
	    float exponentNumerator = -1 * ((x * x) + (y * y));
	    float exponentDenominator = 2 * (sigma * sigma);

	    float e = std::pow(M_E, exponentNumerator / exponentDenominator);
	    kernel.at<float>(x, y) = e / (2 * M_PI * sigma * sigma);
            sum += kernel.at<float>(x, y);
        }
    }

    // Normalize kernel
    for (int i = 0; i < p_kernelSize; ++i) 
    {
        for (int j = 0; j < p_kernelSize; ++j) 
	{
            kernel.at<float>(i,j) /= sum;
        }
    }

    // Apply Gaussian blur
    for (int i = 2; i < rows - 2; ++i)
    {
        for (int j = 2; j < cols - 2; ++j)
	{
            float sum = 0.0;
            for (int m = -2; m <= 2; ++m)
	    {
                for (int n = -2; n <= 2; ++n)
		{
                    sum += inputImage.at<uchar>(i + m, j + n) * kernel.at<float>(m + 2,n + 2);
                }
            }
            blurredImage.at<uchar>(i, j) = sum;
        }
    }

    return blurredImage;
}

Mat applySobelFilter(Mat& gradientX, Mat& gradientY, const Mat& inputImage) 
{
    Mat result;
    Mat abs_grad_x, abs_grad_y;

    // Sobel kernels
    int kernelX[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    int kernelY[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    // Apply Sobel filter in the X direction
    for (int i = 1; i < inputImage.rows - 1; ++i) 
    {
        for (int j = 1; j < inputImage.cols - 1; ++j) 
	{
            for (int m = -1; m <= 1; ++m) 
	    {
                for (int n = -1; n <= 1; ++n) 
		{
                    gradientX.at<float>(i, j) += kernelX[m + 1][n + 1] * inputImage.at<uchar>(i + m, j + n);
                }
            }
        }
    }

    // Apply Sobel filter in the Y direction
    for (int i = 1; i < inputImage.rows - 1; ++i) 
    {
        for (int j = 1; j < inputImage.cols - 1; ++j) 
	{
            for (int m = -1; m <= 1; ++m) 
	    {
                for (int n = -1; n <= 1; ++n) 
		{
                    gradientY.at<float>(i, j) += kernelY[m + 1][n + 1] * inputImage.at<uchar>(i + m, j + n);
                }
            }
        }
    }

    // converting back to CV_8U
    convertScaleAbs(gradientX, abs_grad_x);
    convertScaleAbs(gradientY, abs_grad_y);
    addWeighted(abs_grad_x, 0.2, abs_grad_y, 0.6, 0, result);

    return result;
}

void minimumCutoffSuppression(const cv::Mat& input, cv::Mat& output, int threshold) 
{
    // Apply minimum cutoff suppression by thresholding
    int maxThreshold = 255;
    cv::threshold(input, output, threshold, maxThreshold, cv::THRESH_BINARY);
}

Mat doubleThreshold(const cv::Mat& inputImage, cv::Mat& output, int lowThreshold, int highThreshold) 
{
    Mat result(inputImage.size(), CV_8U, Scalar(0));

    for (int i = 0; i < inputImage.rows; ++i) 
    {
        for (int j = 0; j < inputImage.cols; ++j) 
	{
            uchar gradientMagnitude = inputImage.at<uchar>(i, j);

            if (gradientMagnitude > highThreshold)
	    {
                // Strong edge pixel
                result.at<uchar>(i, j) = 255;
            } 
	    else if (gradientMagnitude < highThreshold && gradientMagnitude > lowThreshold) 
	    {
                // Weak edge pixel
                result.at<uchar>(i, j) = 128;
            } 
	    else if(gradientMagnitude < lowThreshold) 
	    {
                // Non-edge (suppress)
                result.at<uchar>(i, j) = 0;
            }
        }
    }
    return result;
}

int main(int argc, char** argv) 
{
    if(argc < 2)
    {
	std::cerr << "Error: Image not provided." << std::endl;
        return -1;
    }

    int exampleImage[4][10] = { {1, 1, 3, 5, 7, 9, 9, 9, 9, 9},
				{1, 1, 1, 3, 5, 7, 9, 9, 9, 9},
				{1, 1, 1, 1, 3, 5, 7, 9, 9, 9},
				{1, 1, 1, 1, 1, 3, 5, 7, 9, 9} };

    Mat exampleMat = cv::Mat(4, 10, CV_32S, &exampleImage);
    
    // Read the input color image
    CommandLineParser parser( argc, argv, "{@input | input | image}" );
    Mat colorImage = cv::imread( cv::samples::findFile( parser.get<cv::String>( "@input" )));

    if (colorImage.empty()) 
    {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    // Convert color image to grayscale
    Mat grayscaleImage = convertToGrayScale(colorImage);

    // Apply Gaussian Blue
    int kernelSize = 5;
    int sigma = 1;
    Mat blurredImage = applyGaussianBlur(grayscaleImage, kernelSize, sigma);

    // Apply Sobel filterring
    Mat gradientX(blurredImage.rows, blurredImage.cols, CV_32F, Scalar(0));
    Mat gradientY(blurredImage.rows, blurredImage.cols, CV_32F, Scalar(0));
    Mat sobelResult = applySobelFilter(gradientX, gradientY, blurredImage);

    // Apply Lower bound cut-off suppresion
    Mat suppressedEdges;
    int threshold = 100;
    minimumCutoffSuppression(sobelResult, suppressedEdges, threshold);

    // Apply Double threshold
    int lowThreshold = 30;   
    int highThreshold = 100; 

    // Apply double thresholding
    Mat edgesAfterDoubleThreshold = doubleThreshold(suppressedEdges, edgesAfterDoubleThreshold, lowThreshold, highThreshold);

    // Display Canny edge detection steps
    namedWindow("Original Color Image", WINDOW_NORMAL);
    namedWindow("Grayscaled Image", WINDOW_NORMAL);
    namedWindow("Gaussian Blur", WINDOW_NORMAL);
    namedWindow("Sobel filterred Image", WINDOW_NORMAL);
    namedWindow("Suppressed Image", WINDOW_NORMAL);
    namedWindow("Double threshold Image", WINDOW_NORMAL);

    imshow("Original Color Image", colorImage);
    imshow("Grayscaled Image", grayscaleImage);
    imshow("Gaussian Blur", blurredImage);
    imshow("Sobel filterred Image", sobelResult);
    imshow("Suppressed Image", suppressedEdges);
    imshow("Double threshold Image", edgesAfterDoubleThreshold);

    waitKey(0);

    return 0;
}

