#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct Net_config
{
	float confThreshold; // class Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	string modelfile;
};
enum DisType { FR_COSINE = 0, FR_NORM_L2 = 1 };

class YUFace
{
public:
	YUFace(Net_config config);
	Mat detect(Mat frame);
	void setInputSize(const Size& input_size);
private:

	const float stride[3] = { 8.0, 16.0, 32.0 };
	int inputW = 320;
	int inputH = 320;
	float scoreThreshold;
	float nmsThreshold;
	const int topK = 5000;

	void generatePriors();
	Mat postProcess(const vector<Mat>& output_blobs);
	vector<Rect2f> priors;
	Net net;
};

YUFace::YUFace(Net_config config)
{
	this->scoreThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->net = readNet(config.modelfile);
	generatePriors();
}

void YUFace::setInputSize(const Size& input_size)
{
	inputW = input_size.width;
	inputH = input_size.height;
	generatePriors();
}

void YUFace::generatePriors()
{
	// Calculate shapes of different scales according to the shape of input image
	Size feature_map_2nd = {
		int(int((inputW + 1) / 2) / 2), int(int((inputH + 1) / 2) / 2)
	};
	Size feature_map_3rd = {
		int(feature_map_2nd.width / 2), int(feature_map_2nd.height / 2)
	};
	Size feature_map_4th = {
		int(feature_map_3rd.width / 2), int(feature_map_3rd.height / 2)
	};
	Size feature_map_5th = {
		int(feature_map_4th.width / 2), int(feature_map_4th.height / 2)
	};
	Size feature_map_6th = {
		int(feature_map_5th.width / 2), int(feature_map_5th.height / 2)
	};

	vector<Size> feature_map_sizes;
	feature_map_sizes.push_back(feature_map_3rd);
	feature_map_sizes.push_back(feature_map_4th);
	feature_map_sizes.push_back(feature_map_5th);
	feature_map_sizes.push_back(feature_map_6th);

	// Fixed params for generating priors
	const vector<vector<float>> min_sizes = {
		{10.0f,  16.0f,  24.0f},
		{32.0f,  48.0f},
		{64.0f,  96.0f},
		{128.0f, 192.0f, 256.0f}
	};
	const vector<int> steps = { 8, 16, 32, 64 };

	// Generate priors
	priors.clear();
	for (size_t i = 0; i < feature_map_sizes.size(); ++i)
	{
		Size feature_map_size = feature_map_sizes[i];
		vector<float> min_size = min_sizes[i];

		for (int _h = 0; _h < feature_map_size.height; ++_h)
		{
			for (int _w = 0; _w < feature_map_size.width; ++_w)
			{
				for (size_t j = 0; j < min_size.size(); ++j)
				{
					float s_kx = min_size[j] / inputW;
					float s_ky = min_size[j] / inputH;

					float cx = (_w + 0.5f) * steps[i] / inputW;
					float cy = (_h + 0.5f) * steps[i] / inputH;

					Rect2f prior = { cx, cy, s_kx, s_ky };
					priors.push_back(prior);
				}
			}
		}
	}
}

Mat YUFace::postProcess(const vector<Mat>& output_blobs)
{
	// Extract from output_blobs
	Mat loc = output_blobs[0];
	Mat conf = output_blobs[1];
	Mat iou = output_blobs[2];

	// Decode from deltas and priors
	const vector<float> variance = { 0.1f, 0.2f };
	float* loc_v = (float*)(loc.data);
	float* conf_v = (float*)(conf.data);
	float* iou_v = (float*)(iou.data);
	Mat faces;
	// (tl_x, tl_y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score)
	// 'tl': top left point of the bounding box
	// 're': right eye, 'le': left eye
	// 'nt':  nose tip
	// 'rcm': right corner of mouth, 'lcm': left corner of mouth
	Mat face(1, 15, CV_32FC1);
	for (size_t i = 0; i < priors.size(); ++i) {
		// Get score
		float clsScore = conf_v[i * 2 + 1];
		float iouScore = iou_v[i];
		// Clamp
		if (iouScore < 0.f) {
			iouScore = 0.f;
		}
		else if (iouScore > 1.f) {
			iouScore = 1.f;
		}
		float score = sqrt(clsScore * iouScore);
		face.at<float>(0, 14) = score;

		// Get bounding box
		float cx = (priors[i].x + loc_v[i * 14 + 0] * variance[0] * priors[i].width)  * inputW;
		float cy = (priors[i].y + loc_v[i * 14 + 1] * variance[0] * priors[i].height) * inputH;
		float w = priors[i].width  * exp(loc_v[i * 14 + 2] * variance[0]) * inputW;
		float h = priors[i].height * exp(loc_v[i * 14 + 3] * variance[1]) * inputH;
		float x1 = cx - w / 2;
		float y1 = cy - h / 2;
		face.at<float>(0, 0) = x1;
		face.at<float>(0, 1) = y1;
		face.at<float>(0, 2) = w;
		face.at<float>(0, 3) = h;

		// Get landmarks
		face.at<float>(0, 4) = (priors[i].x + loc_v[i * 14 + 4] * variance[0] * priors[i].width)  * inputW;  // right eye, x
		face.at<float>(0, 5) = (priors[i].y + loc_v[i * 14 + 5] * variance[0] * priors[i].height) * inputH;  // right eye, y
		face.at<float>(0, 6) = (priors[i].x + loc_v[i * 14 + 6] * variance[0] * priors[i].width)  * inputW;  // left eye, x
		face.at<float>(0, 7) = (priors[i].y + loc_v[i * 14 + 7] * variance[0] * priors[i].height) * inputH;  // left eye, y
		face.at<float>(0, 8) = (priors[i].x + loc_v[i * 14 + 8] * variance[0] * priors[i].width)  * inputW;  // nose tip, x
		face.at<float>(0, 9) = (priors[i].y + loc_v[i * 14 + 9] * variance[0] * priors[i].height) * inputH;  // nose tip, y
		face.at<float>(0, 10) = (priors[i].x + loc_v[i * 14 + 10] * variance[0] * priors[i].width)  * inputW; // right corner of mouth, x
		face.at<float>(0, 11) = (priors[i].y + loc_v[i * 14 + 11] * variance[0] * priors[i].height) * inputH; // right corner of mouth, y
		face.at<float>(0, 12) = (priors[i].x + loc_v[i * 14 + 12] * variance[0] * priors[i].width)  * inputW; // left corner of mouth, x
		face.at<float>(0, 13) = (priors[i].y + loc_v[i * 14 + 13] * variance[0] * priors[i].height) * inputH; // left corner of mouth, y

		faces.push_back(face);
	}

	if (faces.rows > 1)
	{
		// Retrieve boxes and scores
		vector<Rect2i> faceBoxes;
		vector<float> faceScores;
		for (int rIdx = 0; rIdx < faces.rows; rIdx++)
		{
			faceBoxes.push_back(Rect2i(int(faces.at<float>(rIdx, 0)),
				int(faces.at<float>(rIdx, 1)),
				int(faces.at<float>(rIdx, 2)),
				int(faces.at<float>(rIdx, 3))));
			faceScores.push_back(faces.at<float>(rIdx, 14));
		}

		vector<int> keepIdx;
		NMSBoxes(faceBoxes, faceScores, scoreThreshold, nmsThreshold, keepIdx, 1.f, topK);

		// Get NMS results
		Mat nms_faces;
		for (int idx : keepIdx)
		{
			nms_faces.push_back(faces.row(idx));
		}
		return nms_faces;
	}
	else
	{
		return faces;
	}
}

Mat YUFace::detect(Mat frame)
{
	Mat blob = blobFromImage(frame);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	/////generate proposals
	Mat results = postProcess(outs);
	Mat faces;
	results.convertTo(faces, CV_32FC1);
	return faces;
}

static Mat visualize(Mat input, Mat faces, int thickness = 2)
{
	Mat output = input.clone();
	for (int i = 0; i < faces.rows; i++)
	{
		rectangle(output, Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), Scalar(0, 255, 0), thickness);
		// Draw landmarks
		circle(output, Point2i(int(faces.at<float>(i, 4)), int(faces.at<float>(i, 5))), 2, Scalar(255, 0, 0), thickness);
		circle(output, Point2i(int(faces.at<float>(i, 6)), int(faces.at<float>(i, 7))), 2, Scalar(0, 0, 255), thickness);
		circle(output, Point2i(int(faces.at<float>(i, 8)), int(faces.at<float>(i, 9))), 2, Scalar(0, 255, 0), thickness);
		circle(output, Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2, Scalar(255, 0, 255), thickness);
		circle(output, Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2, Scalar(0, 255, 255), thickness);
	}
	return output;
}

class FaceRecognizer
{
public:
	FaceRecognizer(string modelpath)
	{
		net = readNet(modelpath);
	}
	void alignCrop(InputArray _src_img, InputArray _face_mat, OutputArray _aligned_img)
	{
		Mat face_mat = _face_mat.getMat();
		float src_point[5][2];
		for (int row = 0; row < 5; ++row)
		{
			for (int col = 0; col < 2; ++col)
			{
				src_point[row][col] = face_mat.at<float>(0, row * 2 + col + 4);
			}
		}
		Mat warp_mat = getSimilarityTransformMatrix(src_point);
		warpAffine(_src_img, _aligned_img, warp_mat, Size(112, 112), INTER_LINEAR);
	};
	void feature(InputArray _aligned_img, OutputArray _face_feature)
	{
		Mat inputBolb = blobFromImage(_aligned_img, 1, Size(112, 112), Scalar(0, 0, 0), true, false);
		net.setInput(inputBolb);
		net.forward(_face_feature);
	};
	double match(InputArray _face_feature1, InputArray _face_feature2, int dis_type)
	{
		Mat face_feature1 = _face_feature1.getMat(), face_feature2 = _face_feature2.getMat();
		face_feature1 /= norm(face_feature1);
		face_feature2 /= norm(face_feature2);

		if (dis_type == DisType::FR_COSINE) {
			return sum(face_feature1.mul(face_feature2))[0];
		}
		else if (dis_type == DisType::FR_NORM_L2) {
			return norm(face_feature1, face_feature2);
		}
		else {
			throw invalid_argument("invalid parameter " + to_string(dis_type));
		}

	};
private:
	Net net;
	Mat getSimilarityTransformMatrix(float src[5][2]) const {
		float dst[5][2] = { {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f}, {41.5493f, 92.3655f}, {70.7299f, 92.2041f} };
		float avg0 = (src[0][0] + src[1][0] + src[2][0] + src[3][0] + src[4][0]) / 5;
		float avg1 = (src[0][1] + src[1][1] + src[2][1] + src[3][1] + src[4][1]) / 5;
		//Compute mean of src and dst.
		float src_mean[2] = { avg0, avg1 };
		float dst_mean[2] = { 56.0262f, 71.9008f };
		//Subtract mean from src and dst.
		float src_demean[5][2];
		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				src_demean[j][i] = src[j][i] - src_mean[i];
			}
		}
		float dst_demean[5][2];
		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				dst_demean[j][i] = dst[j][i] - dst_mean[i];
			}
		}
		double A00 = 0.0, A01 = 0.0, A10 = 0.0, A11 = 0.0;
		for (int i = 0; i < 5; i++)
			A00 += dst_demean[i][0] * src_demean[i][0];
		A00 = A00 / 5;
		for (int i = 0; i < 5; i++)
			A01 += dst_demean[i][0] * src_demean[i][1];
		A01 = A01 / 5;
		for (int i = 0; i < 5; i++)
			A10 += dst_demean[i][1] * src_demean[i][0];
		A10 = A10 / 5;
		for (int i = 0; i < 5; i++)
			A11 += dst_demean[i][1] * src_demean[i][1];
		A11 = A11 / 5;
		Mat A = (Mat_<double>(2, 2) << A00, A01, A10, A11);
		double d[2] = { 1.0, 1.0 };
		double detA = A00 * A11 - A01 * A10;
		if (detA < 0)
			d[1] = -1;
		double T[3][3] = { {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0} };
		Mat s, u, vt, v;
		SVD::compute(A, s, u, vt);
		double smax = s.ptr<double>(0)[0] > s.ptr<double>(1)[0] ? s.ptr<double>(0)[0] : s.ptr<double>(1)[0];
		double tol = smax * 2 * FLT_MIN;
		int rank = 0;
		if (s.ptr<double>(0)[0] > tol)
			rank += 1;
		if (s.ptr<double>(1)[0] > tol)
			rank += 1;
		double arr_u[2][2] = { {u.ptr<double>(0)[0], u.ptr<double>(0)[1]}, {u.ptr<double>(1)[0], u.ptr<double>(1)[1]} };
		double arr_vt[2][2] = { {vt.ptr<double>(0)[0], vt.ptr<double>(0)[1]}, {vt.ptr<double>(1)[0], vt.ptr<double>(1)[1]} };
		double det_u = arr_u[0][0] * arr_u[1][1] - arr_u[0][1] * arr_u[1][0];
		double det_vt = arr_vt[0][0] * arr_vt[1][1] - arr_vt[0][1] * arr_vt[1][0];
		if (rank == 1)
		{
			if ((det_u*det_vt) > 0)
			{
				Mat uvt = u * vt;
				T[0][0] = uvt.ptr<double>(0)[0];
				T[0][1] = uvt.ptr<double>(0)[1];
				T[1][0] = uvt.ptr<double>(1)[0];
				T[1][1] = uvt.ptr<double>(1)[1];
			}
			else
			{
				double temp = d[1];
				d[1] = -1;
				Mat D = (Mat_<double>(2, 2) << d[0], 0.0, 0.0, d[1]);
				Mat Dvt = D * vt;
				Mat uDvt = u * Dvt;
				T[0][0] = uDvt.ptr<double>(0)[0];
				T[0][1] = uDvt.ptr<double>(0)[1];
				T[1][0] = uDvt.ptr<double>(1)[0];
				T[1][1] = uDvt.ptr<double>(1)[1];
				d[1] = temp;
			}
		}
		else
		{
			Mat D = (Mat_<double>(2, 2) << d[0], 0.0, 0.0, d[1]);
			Mat Dvt = D * vt;
			Mat uDvt = u * Dvt;
			T[0][0] = uDvt.ptr<double>(0)[0];
			T[0][1] = uDvt.ptr<double>(0)[1];
			T[1][0] = uDvt.ptr<double>(1)[0];
			T[1][1] = uDvt.ptr<double>(1)[1];
		}
		double var1 = 0.0;
		for (int i = 0; i < 5; i++)
			var1 += src_demean[i][0] * src_demean[i][0];
		var1 = var1 / 5;
		double var2 = 0.0;
		for (int i = 0; i < 5; i++)
			var2 += src_demean[i][1] * src_demean[i][1];
		var2 = var2 / 5;
		double scale = 1.0 / (var1 + var2)* (s.ptr<double>(0)[0] * d[0] + s.ptr<double>(1)[0] * d[1]);
		double TS[2];
		TS[0] = T[0][0] * src_mean[0] + T[0][1] * src_mean[1];
		TS[1] = T[1][0] * src_mean[0] + T[1][1] * src_mean[1];
		T[0][2] = dst_mean[0] - scale * TS[0];
		T[1][2] = dst_mean[1] - scale * TS[1];
		T[0][0] *= scale;
		T[0][1] *= scale;
		T[1][0] *= scale;
		T[1][1] *= scale;
		Mat transform_mat = (Mat_<double>(2, 3) << T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2]);
		return transform_mat;
	}
};

int main()
{
	Net_config cfg = { 0.9, 0.3, "weights/face_detection_yunet.onnx" }; 
	YUFace detector(cfg);
	string imgpath = "selfie.jpg";
	Mat srcimg = imread(imgpath);

	detector.setInputSize(srcimg.size());
	Mat faces = detector.detect(srcimg);
	Mat result = visualize(srcimg, faces);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, result);
	waitKey(0);
	destroyAllWindows();

	double cosine_similar_thresh = 0.363;
	double l2norm_similar_thresh = 1.128;
	FaceRecognizer recognizer("weights/face_recognition_sface.onnx");

	string img1path = "telangpu.png";
	string img2path = "telangpu2.png";
	Mat img1 = imread(img1path);
	Mat img2 = imread(img2path);
	
	detector.setInputSize(img1.size());
	Mat faces1 = detector.detect(img1);
	detector.setInputSize(img2.size());
	Mat faces2 = detector.detect(img2);

	Mat aligned_face1, aligned_face2;
	recognizer.alignCrop(img1, faces1.row(0), aligned_face1);
	recognizer.alignCrop(img2, faces2.row(0), aligned_face2);

	Mat feature1, feature2;
	recognizer.feature(aligned_face1, feature1);
	feature1 = feature1.clone();
	recognizer.feature(aligned_face2, feature2);
	feature2 = feature2.clone();

	double cos_score = recognizer.match(feature1, feature2, DisType::FR_COSINE);
	double L2_score = recognizer.match(feature1, feature2, DisType::FR_NORM_L2);
	if (cos_score >= cosine_similar_thresh)
	{
		cout << "They have the same identity;";
	}
	else
	{
		cout << "They have different identities;";
	}
	cout << " Cosine Similarity: " << cos_score << ", threshold: " << cosine_similar_thresh << ". (higher value means higher similarity, max 1.0)\n";

	if (L2_score <= l2norm_similar_thresh)
	{
		cout << "They have the same identity;";
	}
	else
	{
		cout << "They have different identities.";
	}
	cout << " NormL2 Distance: " << L2_score << ", threshold: " << l2norm_similar_thresh << ". (lower value means higher similarity, min 0.0)\n";
}