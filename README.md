# Gaze-Tracking
face detection(landmarks), head pose estimation, gaze estimation. 
## appearance-based methods  
### 0、Review
#### • Appearance-based Gaze Estimation With Deep Learning：A Review and Benchmark(arxiv2021)  
#### • Automatic Gaze Analysis：A Survey of Deep Learning based Approaches(arxiv2021)  
#### • A survey on Deep Learning Based Eye Gaze Estimation Methods（Journal of Innovative Image Processing 2021)  
#### • Evaluation of appearance-based methods and implications for gaze-based applications (zhang19_chi)

### 1、General 
#### •  Learning-by-Synthesis for Appearance-based 3D Gaze Estimation(东京大学CVPR2014)
#### •  M-norm_Learning-by-Synthesis for Appearance-Based 3D Gaze Estimation（cvpr2014）
#### •  Where are they looking(nips2015)
#### •  Gaze Estimation from Eye Appearance A Head Pose-Free Method via Eye Image Synthesis(TIP2015)
#### •  Person-specific：Appearance-based gaze estimation in the wild（cvpr2015headpose拼接）
#### •  Rendering of eyes for eye-shape registration and gaze estimation(ICCV2015 norm)
#### •  Highly Accurate Gaze Estimation Using a Consumer RGB-D Sensor（arxiv2016）
#### •  Eye Tracking for Everyone(cvpr2016)
#### •  It’s Written All Over Your Face：Full-Face Appearance-Based Gaze Estimation(全脸CVPRW2017)
#### •  Monocular free-head 3d gaze tracking with deep learning and geometry constraints（头部同步估计ICCV2017商汤）
#### •  M-norm_MPIIGaze： Real-World Dataset and Deep Appearance-Based Gaze Estimation(tpami2017)
#### •  Appearance-Based Gaze Estimation Using Dilated-Convolutions(ACCV2018)
#### •  ARNet：Appearance-based gaze estimation via evaluation- guided asymmetric regression(ECCV2018北航)
#### •  RT-GENE：Real-Time Eye Gaze Estimation in Natural Environments(头部同步估计 ECCV2018)
#### •  Deep Pictorial Gaze Estimation（eccv2018）
#### •  Efficient CNN Implementation for Eye-Gaze Estimation on Low-Power Low-Quality Consumer Imaging Systems (arxiv2018)
#### •  R-norm:revisiting data normlization for appearance-based gaze estimation（etra2018重要）
#### •  Recurrent CNN for 3D Gaze Estimation using Appearance and Shape Cues（BMVC2018）
#### •  Unconstrained and Calibration-Free Gaze Estimation in a Room-Scale Area Using a Monocular Camera（ACCESS2018）
#### •  A Generalized and Robust Method Towards Practical Gaze Estimation on Smart Phone(ICCVW2019)
#### •  Dilated-Net：Appearance-Based Gaze Estimation Using Dilated-Convolutions(ACCV2019)
#### •  Gaze360：Physically Unconstrained Gaze Estimation in the Wild(ICCV2019)
#### •  Generalizing Eye Tracking With Bayesian Adversarial Learning（cvpr2019）
#### •  RGBD Based Gaze Estimation via Multi-Task CNN(AAAI2019)
#### •  NISLGaze：Unsupervised Outlier Detection in Appearance-Based Gaze Estimation(HKUST_ICCVW2019)
#### •  RT-BENE：A Dataset and Baselines for Real-Time Blink Estimation in Natural Environments（视线头部眨眼估计ICCVW2019）
#### •  EVE：Towards End-to-end Video-based Eye-Tracking（eccv2020）
#### •  Gaze estimation problem tackled through synthetic images(arxiv2020)
#### •  Efficiency in Real-time Webcam  Gaze Tracking(ECCV2020)
#### •  AFFNet:Adaptive Feature Fusion Network for Gaze Tracking in Mobile Tablets(ICPR2020poster)
#### •  R-norm:ETH-XGaze：A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation(ECCV2020)
#### •  Robot Navigation in Crowds by Graph Convolutional Networks With Attention Learned From Human Gaze（icra2020_final）
#### •  A Coarse-to-fine Adaptive Network for Appearance-based Gaze Estimation（AAAI2020）
#### •  Unsupervised Representation Learning for Gaze Estimation(CVPR2020)
#### •  Self-Learning Transformations for Improving Gaze and Head Redirection(NIPS2020)
#### •  Accurate Pupil Center Detection in Off-the-Shelf Eye Tracking Systems Using Convolutional Neural Networks(sensors2021)
#### •  Appearance-based Gaze Estimation using Attention and Difference Mechanism（cvprw2021）
#### •  Gaze Estimation with an Ensemble of Four Architectures(CVPR2021 GAZE challenge)
#### •  Looking Here or There Gaze Following in 360-DegreeImages(ICCV2021)
#### •  Transformer：Gaze Estimation using Transformer(2021arxivV1 beihang)
#### •  Weakly-Supervised Physically Unconstrained Gaze Estimation（cvpr2021）
#### •  Bayesian_Eye_Tracking(arxiv2021)
#### •  pnp-GA：Generalizing Gaze Estimation with Outlier-guided Collaborative Adaptation(ICCV2021)
#### •  TEyeD：Over 20 million real-world eye images with Pupil, Eyelid, and Iris 2D and 3D Segmentations, 2D and 3D Landmarks, 3D Eyeball, Gaze Vector, and Eye Movement Types（arxiv2021）
#### •  L2CS-Net：Fine-Grained Gaze Estimation in Unconstrained Environments（arxivV1 2022）
#### •  MTGLS：Multi-Task Gaze Estimation With Limited Supervision(WACV2022)
#### •  PureGaze：Purifying Gaze Feature for Generalizable Gaze Estimation(AAAI2022beihang)

### 2、Few-Shot
#### •  SAGE：On-device Few-shot Personalization for Real-time Gaze Estimation(ICCVW2019google)
#### •  GRS：Gaze Redirection：Improving few-shot user-specific gaze adaptation via gaze redirection synthesis（CVPR2019）
#### •  FAZE：Few-Shot Adaptive Gaze Estimation(ICCV2019oral)
#### •  PupilTAN：A Few-Shot Adversarial Pupil Localizer（cvprw2021）

### 3、Calibration  
#### • A Regression-based User Calibration Framework for real-time gaze estimation（TCSVT2016）
#### • A Statistical Approach to Continuous Self-Calibrating Eye Gaze Tracking for Head-Mounted Virtual Reality Systems(Applications of Computer Vision2017)
#### • Auto-Calibrated Gaze Estimation Using Human Gaze Patterns（IJCV2017）
#### • A Robust Extrinsic Calibration Method for Non-Contact Gaze Tracking in the 3-D Space(ACCESS2018)
#### • Kappa：Positions of Ocular Geometrical and Visual Axes in Brazilian, Chinese and Italian Populations（2018）
#### • Low Cost Gaze Estimation Knowledge-Based Solutions(tip2019)
#### • SPAZE：Learning to Personalize in Appearance-Based Gaze Tracking(ICCVW2019)
#### • GD：Offset Calibration for Appearance-Based Gaze Estimation via Gaze Decomposition（WACV2020）
#### • Gaze-Estimation-via-a-Differential-Eyes--Appearances-Network-with a Reference Grid(Engineering2021)
#### • A Differential Approach for Gaze Estimation(TPAMI2021)
#### • Low-Cost Eye Tracking Calibration A Knowledge-Based Study(sensors2021)
#### • Learning-by-Novel-View-Synthesis for Full-Face Appearance-Based 3D Gaze Estimation（arxivV3-cvpr2022）
#### • GEDDnet：Towards High Performance Low Complexity Calibration in Appearance Based Gaze Estimation（TPAMI2022）
#### • Resolving Camera Position for a Practical Application of Gaze Estimation on Edge Devices(arxiv2201.02946v2)

## otherwise 
#### •  单相机无红外：3D Gaze Estimation with a Single Camera without IR Illumination（ICPR2008）
#### •  RGBD相机：Eye Gaze Tracking Using an RGBD Camera：A Comparison with an RGB Solution（UbiComp2014）
#### •  消费级深度相机：Real time gaze estimation with a consumer depth camera(INS2015)
#### •  Real Time Eye Gaze Tracking with Kinect（ICPR2016）
#### •  几何模型：Real time eye gaze tracking with 3d deformable eye-face model（ICCV2017）
#### •  深度相机：Two-eye model-based Gaze Estimation from A Kinect Sensor(ICRA2017)
#### •  kappa标定：三维视线追踪系统中Kappa角标定问题研究（北京科技大学2018）
#### •  3D gaze estimation without explicit personal calibration（pattern recognition2018）
#### •  MeNets：Mixed effects neural networks (menets) with applications to gaze estimation（CVPR2019）
#### •  车载ADAS：Gaze and Eye Tracking：Techniques and Applications in ADAS（sensors2019）
#### •  基于简化硬件系统的三维视线估计方法研究（北京科技大学博士论文2020）
#### •  标定（单相机多光源）：Screen-Light Decomposition Framework for Point-of-Gaze Estimation Using a Single Uncalibrated Camera and Multiple Light Sources(Morimoto2020)
#### •  综述：基于特征的视线跟踪方法研究综述（自动化学报2021）
#### •  内插法：High-Accuracy Gaze Estimation for Interpolation-Based Eye-Tracking Methods(vision2021)
#### •  眼动跟踪研究进展与展望（自动化学报2022）

