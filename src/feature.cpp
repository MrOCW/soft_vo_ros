#include "feature.h"
#include "bucket.h"
cv::Mat blobFilter = (cv::Mat_<float>(5, 5) << -1,-1,-1,-1,-1,-1,1,1,1,-1,-1,1,8,1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1);
cv::Mat cornerFilter = (cv::Mat_<float>(5, 5) << -1,-1,0,1,1,-1,-1,0,1,1,0,0,0,0,0,1,1,0,-1,-1,1,1,0,-1,-1);

#if USE_CUDA
static void download(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

static void download(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}


bool checkMinValidity(cv::Mat& inputImg, int minPixIntensity, int minI, int minJ, int n, int margin)
{
  int height = inputImg.size().height;
  int width = inputImg.size().width;
  double min;
  cv::Point minLoc;
  cv::Mat window = inputImg(cv::Range((minI - n),std::min(minI + n, height - 1 - margin)),cv::Range((minJ - n),std::min(minJ + n, width - 1 - margin)));
  cv::minMaxLoc(window,&min,NULL,&minLoc);
  //cv::imshow("check",window);
  //cv::waitKey(0);
  if (min < minPixIntensity && ( minLoc.x < minJ || minLoc.x > (minJ + n) || minLoc.y < minI || minLoc.y > (minI + n)))
  {
    return true;
  }
  else
  {
    return false;
  }
}

bool checkMaxValidity(cv::Mat& inputImg, int maxPixIntensity, int maxI, int maxJ, int n, int margin)
{
  int height = inputImg.size().height;
  int width = inputImg.size().width;
  double max;
  cv::Point maxLoc;
  cv::Mat window = inputImg(cv::Range((maxI - n),std::min(maxI + n, height - 1 - margin)),cv::Range((maxJ - n),std::min(maxJ + n, width - 1 - margin)));
  cv::minMaxLoc(window,NULL,&max,NULL,&maxLoc);
  if (max > maxPixIntensity && ( maxLoc.x < maxJ|| maxLoc.x > (maxJ + n) || maxLoc.y < maxI || maxLoc.y > (maxI + n)))
  {
    return true;
  }
  else
  {
    return false;
  }
}

void nms(cv::Mat& blobImg,cv::Mat& cornerImg,FeatureSet& voFeatures,int n=8, int tau=50, int margin=10)
{
  
  int height = blobImg.size().height;
  int width = blobImg.size().width;
  //std::cout<<"Height:"<<height<<std::endl;
  //std::cout<<"Width:"<<width<<std::endl;

  int blobMin_i;
  int blobMin_j;
  int blobMax_i;
  int blobMax_j;
  int cornerMin_i;
  int cornerMin_j;
  int cornerMax_i;
  int cornerMax_j;

  int blobMin;
  int blobMax;
  int cornerMin;
  int cornerMax;

  int currval;
  //int count = 0;
  
  for (int i=(n+margin);i<=(height - n - margin);i+=(n+1))
  {
    
    for (int j=(n+margin);j<=(width-n-margin);j+=(n+1))
    {
      //std::cout<<count<<std::endl;
      //std::cout<<i<<" "<<j<<std::endl;
      //cv::Mat window = blobImg(cv::Range(i,i+50),cv::Range(j,j+100));
      //cv::imshow("Window location",window);

      blobMin_i = i;
      blobMin_j = j;
      blobMax_i = i;
      blobMax_j = j;
      cornerMin_i = i;
      cornerMin_j = j;
      cornerMax_i = i;
      cornerMax_j = j;
      
      blobMin = blobImg.at<float>(i,j);
      blobMax = blobMin;
      cornerMin = cornerImg.at<float>(i,j);
      cornerMax = cornerMin;
      // loop through this window to get min and max values
      for (int i2=i;i2<=(i + n);i2++)
      {
        for (int j2=j;j2<=(j + n);j2++)
        {
          //blob detector
          currval = blobImg.at<float>(i2, j2);
          if (currval < blobMin)
          {
            blobMin_i = i2;
            blobMin_j = j2;
            blobMin = currval;
          }         
          else if (currval > blobMax)
          {
            blobMax_i = i2;
            blobMax_j = j2;
            blobMax = currval;
          }

          //corner detector
          currval = cornerImg.at<float>(i2, j2);
          if (currval < cornerMin)
          {
            cornerMin_i = i2;
            cornerMin_j = j2;
            cornerMin = currval;
          }

          else if (currval > cornerMax)
          {
            cornerMax_i = i2;
            cornerMax_j = j2;
            cornerMax = currval;
          } 
        }       
      }

      if (!checkMinValidity(blobImg, blobMin, blobMin_i, blobMin_j, n, margin))
      {
        if (blobMin <= -tau)
        {
          // FeaturePoint minBlobPoint;
          // minBlobPoint.point = cv::Point2f(blobMin_i,blobMin_j);
          // minBlobPoint.age = 0;
          // minBlobPoint.featureClass = 1;
          // minBlobPoint.value = blobMin;
          // voFeatures.points.push_back(minBlobPoint);
          voFeatures.points.push_back(cv::Point2f(blobMin_j,blobMin_i));
          voFeatures.ages.push_back(0);

        }
      }
      if (!checkMaxValidity(blobImg, blobMax, blobMax_i, blobMax_j, n, margin))
      {
        if (blobMax >= tau)
        {
          // FeaturePoint maxBlobPoint;
          // maxBlobPoint.point = cv::Point2f(blobMax_i,blobMax_j);
          // maxBlobPoint.age = 0;
          // maxBlobPoint.featureClass = 2;
          // maxBlobPoint.value = blobMax;
          // voFeatures.points.push_back(maxBlobPoint);
          voFeatures.points.push_back(cv::Point2f(blobMax_j,blobMax_i));
          voFeatures.ages.push_back(0);
        }
      }
      if (!checkMinValidity(cornerImg, cornerMin, cornerMin_i, cornerMin_j, n, margin))
      {
        if (cornerMin <= -tau)
        {
          // FeaturePoint minCornerPoint;
          // minCornerPoint.point = cv::Point2f(cornerMin_i,cornerMin_j);
          // minCornerPoint.age = 0;
          // minCornerPoint.featureClass = 3;
          // minCornerPoint.value = cornerMin;
          // voFeatures.points.push_back(minCornerPoint);
          voFeatures.points.push_back(cv::Point2f(cornerMin_j,cornerMin_i));
          voFeatures.ages.push_back(0);
        }
      }
      if (!checkMaxValidity(cornerImg, cornerMax, cornerMax_i, cornerMax_j, n, margin))
      {
        if (cornerMax >= tau)
        {
          // FeaturePoint maxCornerPoint;
          // maxCornerPoint.point = cv::Point2f(cornerMax_i,cornerMax_j);
          // maxCornerPoint.age = 0;
          // maxCornerPoint.featureClass = 3;
          // maxCornerPoint.value = cornerMin;
          // voFeatures.points.push_back(maxCornerPoint);
          voFeatures.points.push_back(cv::Point2f(cornerMax_j,cornerMax_i));
          voFeatures.ages.push_back(0);
        }
      }
    }
  }
  //std::cout<<voFeatures.size()<<std::endl;
  //std::cout<<"NMS Complete"<<std::endl;
}

void featureDetection(cv::Mat& inputImg,FeatureSet& voFeatures)
{
  // cv::Mat blobFeatureImgCV32;
  // cv::Mat cornerFeatureImgCV32;
  cv::Mat blobFeatureImg;
  cv::Mat cornerFeatureImg;
  cv::Mat inputImgCV32;
  inputImg.convertTo(inputImgCV32,5);
  cv::cuda::GpuMat inputImgGpu(inputImgCV32);
  cv::cuda::GpuMat blobFeatureImgGpu;
  cv::cuda::GpuMat cornerFeatureImgGpu;
  cv::Ptr<cv::cuda::Convolution> convolver = cv::cuda::createConvolution(cv::Size(5, 5));
  convolver->convolve(inputImgGpu, blobFilter, blobFeatureImgGpu);
  convolver->convolve(inputImgGpu, cornerFilter, cornerFeatureImgGpu);
  blobFeatureImgGpu.download(blobFeatureImg); // this is FP32!
  cornerFeatureImgGpu.download(cornerFeatureImg);
  // blobFeatureImgCV32.convertTo(blobFeatureImg,0);
  // cornerFeatureImgCV32.convertTo(cornerFeatureImg,0);
  nms(blobFeatureImg,cornerFeatureImg,voFeatures);
  // cv::imshow("Blob Features",blobFeatureImg);
  // cv::imshow("Corner Features",cornerFeatureImg);
  // cv::imshow("Original",inputImg);
  //drawFeaturePoints(inputImg,voFeatures.points,2);
  //cv::imshow("Post-NMS",inputImg);
  //cv::waitKey(0);
}
#endif
void deleteUnmatchFeatures(std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1, std::vector<uchar>& status)
{
  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  cv::Point2f pt = points1.at(i- indexCorrection);
        if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))   
        {
              if((pt.x<0)||(pt.y<0))    
              {
                status.at(i) = 0;
              }
              points0.erase (points0.begin() + (i - indexCorrection));
              points1.erase (points1.begin() + (i - indexCorrection));
              indexCorrection++;
        }
     }
}

void featureDetectionFast(cv::Mat image, std::vector<cv::Point2f>& points)  
{   
//uses FAST as for feature dection, modify parameters as necessary
  std::vector<cv::KeyPoint> keypoints;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  cv::FAST(image, keypoints, fast_threshold, nonmaxSuppression);
  cv::KeyPoint::convert(keypoints, points, std::vector<int>());
}

void featureDetectionGoodFeaturesToTrack(cv::Mat image, std::vector<cv::Point2f>& points)  
{   
//uses GoodFeaturesToTrack for feature dection, modify parameters as necessary

  int maxCorners = 5000;
  double qualityLevel = 0.01;
  double minDistance = 5.;
  int blockSize = 3;
  bool useHarrisDetector = false;
  double k = 0.04;
  cv::Mat mask;

  cv::goodFeaturesToTrack( image, points, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k );
}

void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, std::vector<uchar>& status)  
{ 
//this function automatically gets rid of points for which tracking fails

  std::vector<float> err;                    
  cv::Size winSize=cv::Size(21,21);                                                                                             
  cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
  deleteUnmatchFeatures(points1, points2, status);
}

void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1,
                          std::vector<cv::Point2f>& points2, std::vector<cv::Point2f>& points3,
                          std::vector<cv::Point2f>& points0_return,
                          std::vector<uchar>& status0, std::vector<uchar>& status1,
                          std::vector<uchar>& status2, std::vector<uchar>& status3,
                          std::vector<int>& ages){
  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  for (int i = 0; i < ages.size(); ++i)
  {
     ages[i] += 1;
  }

  int indexCorrection = 0;
  for( int i=0; i<status3.size(); i++)
     {  cv::Point2f pt0 = points0.at(i- indexCorrection);
        cv::Point2f pt1 = points1.at(i- indexCorrection);
        cv::Point2f pt2 = points2.at(i- indexCorrection);
        cv::Point2f pt3 = points3.at(i- indexCorrection);
        cv::Point2f pt0_r = points0_return.at(i- indexCorrection);
        
        if ((status3.at(i) == 0)||(pt3.x<0)||(pt3.y<0)||
            (status2.at(i) == 0)||(pt2.x<0)||(pt2.y<0)||
            (status1.at(i) == 0)||(pt1.x<0)||(pt1.y<0)||
            (status0.at(i) == 0)||(pt0.x<0)||(pt0.y<0))   
        {
          if((pt0.x<0)||(pt0.y<0)||(pt1.x<0)||(pt1.y<0)||(pt2.x<0)||(pt2.y<0)||(pt3.x<0)||(pt3.y<0))    
          {
            status3.at(i) = 0;
          }
          points0.erase (points0.begin() + (i - indexCorrection));
          points1.erase (points1.begin() + (i - indexCorrection));
          points2.erase (points2.begin() + (i - indexCorrection));
          points3.erase (points3.begin() + (i - indexCorrection));
          points0_return.erase (points0_return.begin() + (i - indexCorrection));

          ages.erase (ages.begin() + (i - indexCorrection));
          indexCorrection++;
        }

     }  
}

void circularMatching(cv::Mat img_l_0, cv::Mat img_r_0, cv::Mat img_l_1, cv::Mat img_r_1,
                      std::vector<cv::Point2f>& points_l_0, std::vector<cv::Point2f>& points_r_0,
                      std::vector<cv::Point2f>& points_l_1, std::vector<cv::Point2f>& points_r_1,
                      std::vector<cv::Point2f>& points_l_0_return,
                      FeatureSet& current_features) { 
  
  //this function automatically gets rid of points for which tracking fails

  std::vector<float> err;                    
  cv::Size winSize=cv::Size(21,21);                                                                                             
  cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);

  std::vector<uchar> status0;
  std::vector<uchar> status1;
  std::vector<uchar> status2;
  std::vector<uchar> status3;

  clock_t tic = clock();

  calcOpticalFlowPyrLK(img_l_0, img_r_0, points_l_0, points_r_0, status0, err, winSize, 3, termcrit, 0, 0.001);
  calcOpticalFlowPyrLK(img_r_0, img_r_1, points_r_0, points_r_1, status1, err, winSize, 3, termcrit, 0, 0.001);
  calcOpticalFlowPyrLK(img_r_1, img_l_1, points_r_1, points_l_1, status2, err, winSize, 3, termcrit, 0, 0.001);
  calcOpticalFlowPyrLK(img_l_1, img_l_0, points_l_1, points_l_0_return, status3, err, winSize, 3, termcrit, 0, 0.001);
  clock_t toc = clock();
  std::cerr << "calcOpticalFlowPyrLK time: " << float(toc - tic)/CLOCKS_PER_SEC*1000 << "ms" << std::endl;


  deleteUnmatchFeaturesCircle(points_l_0, points_r_0, points_r_1, points_l_1, points_l_0_return,
                        status0, status1, status2, status3, current_features.ages);

  // std::cout << "points : " << points_l_0.size() << " "<< points_r_0.size() << " "<< points_r_1.size() << " "<< points_l_1.size() << " "<<std::endl;
}

#if USE_CUDA

void circularMatching_gpu(cv::Mat img_l_0, cv::Mat img_r_0, cv::Mat img_l_1, cv::Mat img_r_1,
                      std::vector<cv::Point2f>& points_l_0, std::vector<cv::Point2f>& points_r_0,
                      std::vector<cv::Point2f>& points_l_1, std::vector<cv::Point2f>& points_r_1,
                      std::vector<cv::Point2f>& points_l_0_return,
                      FeatureSet& current_features) { 
  //this function automatically gets rid of points for which tracking fails

  cv::Size winSize=cv::Size(21,21);                                                                                             

  std::vector<uchar> status0;
  std::vector<uchar> status1;
  std::vector<uchar> status2;
  std::vector<uchar> status3;
  
  clock_t tic_gpu = clock();
  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
            winSize, 3, 30);
  
  cv::cuda::GpuMat img_l_0_gpu(img_l_0);
  cv::cuda::GpuMat img_r_0_gpu(img_r_0);
  cv::cuda::GpuMat img_l_1_gpu(img_l_1);
  cv::cuda::GpuMat img_r_1_gpu(img_r_1);
  cv::cuda::GpuMat status0_gpu(status0);
  cv::cuda::GpuMat status1_gpu(status1);
  cv::cuda::GpuMat status2_gpu(status2);
  cv::cuda::GpuMat status3_gpu(status3);
  cv::cuda::GpuMat points_l_0_gpu(points_l_0);
  cv::cuda::GpuMat points_r_0_gpu(points_r_0);
  cv::cuda::GpuMat points_l_1_gpu(points_l_1);
  cv::cuda::GpuMat points_r_1_gpu(points_r_1);
  cv::cuda::GpuMat points_l_0_ret_gpu(points_l_0_return);

  d_pyrLK_sparse->calc(img_l_0_gpu, img_r_0_gpu, points_l_0_gpu, points_r_0_gpu, status0_gpu);
  d_pyrLK_sparse->calc(img_r_0_gpu, img_r_1_gpu, points_r_0_gpu, points_r_1_gpu, status1_gpu);
  d_pyrLK_sparse->calc(img_r_1_gpu, img_l_1_gpu, points_r_1_gpu, points_l_1_gpu, status2_gpu);
  d_pyrLK_sparse->calc(img_l_1_gpu, img_l_0_gpu, points_l_1_gpu, points_l_0_ret_gpu, status3_gpu);

  download(status0_gpu, status0);
  download(status1_gpu, status1);
  download(status2_gpu, status2);
  download(status3_gpu, status3);
  download(points_l_0_gpu, points_l_0);
  download(points_l_1_gpu, points_l_1);
  download(points_r_0_gpu, points_r_0);
  download(points_r_1_gpu, points_r_1);
  download(points_l_0_ret_gpu, points_l_0_return);
  
  clock_t toc_gpu = clock();
  std::cerr << "calcOpticalFlowPyrLK(CUDA)  time: " << float(toc_gpu - tic_gpu)/CLOCKS_PER_SEC*1000 << "ms" << std::endl;

  deleteUnmatchFeaturesCircle(points_l_0, points_r_0, points_r_1, points_l_1, points_l_0_return,
                        status0, status1, status2, status3, current_features.ages);
}
#endif

void bucketingFeatures(cv::Mat& image, FeatureSet& current_features, int bucket_size, int features_per_bucket)
{
// This function buckets features
// image: only use for getting dimension of the image
// bucket_size: bucket size in pixel is bucket_size*bucket_size
// features_per_bucket: number of selected features per bucket
    int image_height = image.rows;
    int image_width = image.cols;
    int buckets_nums_height = image_height/bucket_size;
    int buckets_nums_width = image_width/bucket_size;
    int buckets_number = buckets_nums_height * buckets_nums_width;
    std::cout << "buckets_number: " <<buckets_number << std::endl;

    std::vector<Bucket> Buckets;

    // initialize all the buckets
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
      for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
      {
        Buckets.push_back(Bucket(features_per_bucket));
      }
    }

    // bucket all current features into buckets by their location
    int buckets_nums_height_idx, buckets_nums_width_idx, buckets_idx;
    for (int i = 0; i < current_features.points.size(); ++i)
    {
      buckets_nums_height_idx = current_features.points[i].y/bucket_size;
      buckets_nums_width_idx = current_features.points[i].x/bucket_size;
      buckets_idx = buckets_nums_height_idx*buckets_nums_width + buckets_nums_width_idx;
      Buckets[buckets_idx].add_feature(current_features.points[i], current_features.ages[i]);

    }

    // get features back from buckets
    current_features.clear();
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
      for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
      {
         buckets_idx = buckets_idx_height*buckets_nums_width + buckets_idx_width;
         Buckets[buckets_idx].get_features(current_features);
      }
    }

    std::cout << "current features number after bucketing: " << current_features.size() << std::endl;

}

void appendNewFeatures(cv::Mat& image, FeatureSet& current_features)
{
    std::vector<cv::Point2f>  points_new;
    featureDetectionFast(image, points_new);
    //featureDetectionGoodFeaturesToTrack(image,points_new);
    current_features.points.insert(current_features.points.end(), points_new.begin(), points_new.end());
    std::vector<int>  ages_new(points_new.size(), 0);
    current_features.ages.insert(current_features.ages.end(), ages_new.begin(), ages_new.end());
}

void appendNewFeatures(std::vector<cv::Point2f> points_new, FeatureSet& current_features)
{
    current_features.points.insert(current_features.points.end(), points_new.begin(), points_new.end());
    std::vector<int>  ages_new(points_new.size(), 0);
    current_features.ages.insert(current_features.ages.end(), ages_new.begin(), ages_new.end());
}
