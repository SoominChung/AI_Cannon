
//------------------------------------------------------------------------------------------------
// File: Cannon.cpp
// Project: LG Exec Ed Program
// Versions:
// 1.0 April 2024 - initial version
//------------------------------------------------------------------------------------------------
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>
#include <stdio.h>
#include <signal.h>
#include <pthread.h>
#include <sys/select.h>
#include "NetworkTCP.h"
#include "TcpSendRecvJpeg.h"
#include "Message.h"
#include "KeyboardSetup.h"
#include "IsRPI.h"
#include "ServoPi.h"
#include "ObjectDetector.h"
#include <jetgpio.h>
#include "CvImageMatch.h"
#include "ssd1306.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <unistd.h>
#include <vector>
#include <atomic>      // std::atomic 사용을 위해
#include <thread>      // std::this_thread 사용을 위해

#define USE_IMAGE_MATCH 1

#define PORT            5000
#define PAN_SERVO       1
#define TILT_SERVO      2
#define MIN_TILT         (-35.0f)
#define MAX_TILT         ( 35.0f)
#define MIN_PAN          (-85.0f)
#define MAX_PAN          ( 85.0f)


#define WIDTH           1920
#define HEIGHT          1080

#define INC             0.5f

#define USE_USB_WEB_CAM 0

#define SHARED_MEMORY_NAME "/object_detection"
#define MEMORY_SIZE 1024

#define STATIONARY_SHOOTING 0

using namespace cv;
using namespace std;


typedef enum
{
 NOT_ACTIVE,
 ACTIVATE,
 NEW_TARGET,
 LOOKING_FOR_TARGET,
 TRACKING,
 TRACKING_STABLE,
 ENGAGEMENT_IN_PROGRESS,
 ENGAGEMENT_COMPLETE
} TEngagementState;


typedef struct
{
 int                       NumberOfTartgets;
 int                       FiringOrder[10];
 int                       CurrentIndex;
 bool                      HaveFiringOrder;
 volatile TEngagementState State;
 int                       StableCount;
 float                     LastPan;
 float                     LastTilt;
 int                       Target;
} TAutoEngage;

// 새로운 thread를 위한 구조체
struct FireControl {
    std::atomic<bool> is_firing{false};
    std::chrono::steady_clock::time_point last_fire_time;

    #if STATIONARY_SHOOTING == 1
        static constexpr int FIRE_DURATION_MS = 50;
    #else
        static constexpr int FIRE_DURATION_MS = 400;
    #endif    
};

FireControl fireControl;


static TAutoEngage            AutoEngage;
static float                  Pan=0.0f;
static float                  Tilt=0.0f;
static unsigned char          RunCmds=0;
static uint8_t                i2c_node_address = 1;
static bool                   HaveOLED=false;
static int                    OLED_Font=0;
static pthread_t              NetworkThreadID=-1;
static pthread_t              EngagementThreadID=-1;
static volatile SystemState_t SystemState= SAFE;
static pthread_mutex_t        TCP_Mutex;
static pthread_mutex_t        GPIO_Mutex;
static pthread_mutex_t        I2C_Mutex;
static pthread_mutex_t        Engmnt_Mutex;
static pthread_mutexattr_t    TCP_MutexAttr;
static pthread_mutexattr_t    GPIO_MutexAttr;
static pthread_mutexattr_t    I2C_MutexAttr;
static pthread_mutexattr_t    Engmnt_MutexAttr;
static pthread_cond_t         Engagement_cv;
static float                  xCorrect=60.0,yCorrect=-90.0;
static volatile bool          isConnected=false;
static Servo                  *Servos=NULL;
cv::VideoCapture              *capture=NULL;
static Mat                    NoDataAvalable;
static TTcpListenPort         *TcpListenPort=NULL;
static TTcpConnectedPort      *TcpConnectedPort=NULL;
static int                    capture_width = 1280 ;
static int                    capture_height = 720 ;
static int                    display_width =  capture_width ;
static int                    display_height = capture_height ;
static int                    framerate = 30 ;
static int                    flip_method = 0 ;

static void   Setup_Control_C_Signal_Handler_And_Keyboard_No_Enter(void);
static void   CleanUp(void);
static void   Control_C_Handler(int s);
static void   HandleInputChar();
static void * NetworkInputThread(void *data);
static void * EngagementThread(void *data); 
static int    PrintfSend(const char *fmt, ...); 
static bool   GetFrame( Mat &frame);
static void   CreateNoDataAvalable(void);
static int    SendSystemState(SystemState_t State);
static bool   compare_float(float x, float y, float epsilon = 0.5f);
static void   ServoAngle(int Num,float &Angle) ;
static void * EngagementCustom(void);

void printBoundingBoxes(const std::vector<BoundingBox>& bboxes);

/*************************************** TF LITE START ********************************************************/ 
#if USE_TFLITE && !USE_IMAGE_MATCH
static ObjectDetector *detector;
/*************************************** TF LITE END   ********************************************************/ 
#elif USE_IMAGE_MATCH && !USE_TFLITE
/*************************************** IMAGE_MATCH START *****************************************************/ 


/*************************************** IMAGE_MATCH END *****************************************************/ 
#endif

//------------------------------------------------------------------------------------------------
// static std::string gstreamer_pipeline
//------------------------------------------------------------------------------------------------
std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
#if 0
return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
#else
return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
#endif
}
//------------------------------------------------------------------------------------------------
// END static std::string gstreamer_pipeline
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void ReadOffsets
//------------------------------------------------------------------------------------------------
static void ReadOffsets(void)
{
   FILE * fp;
   float x,y;
   char xs[100],ys[100];
   int retval=0;

   fp = fopen ("Correct.ini", "r");
   retval+=fscanf(fp, "%s %f", xs,&x);
   retval+=fscanf(fp, "%s %f", ys,&y);
   if (retval==4)
   {
    if ((strcmp(xs,"xCorrect")==0) && (strcmp(ys,"yCorrect")==0))
       {
         xCorrect=x;
         yCorrect=y;
         printf("Read Offsets:\n");
         printf("xCorrect= %f\n",xCorrect);
         printf("yCorrect= %f\n",yCorrect);
       }
   }
   fclose(fp);

}
//------------------------------------------------------------------------------------------------
// END  static void readOffsets
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void readOffsets
//------------------------------------------------------------------------------------------------
static void WriteOffsets(void)
{
   FILE * fp;
   float x,y;
   char xs[100],ys[100];
   int retval=0;

   fp = fopen ("Correct.ini", "w+");
   rewind(fp);
   fprintf(fp,"xCorrect %f\n", xCorrect);
   fprintf(fp,"yCorrect %f\n", yCorrect);
      
   printf("Wrote Offsets:\n");
   printf("xCorrect= %f\n",xCorrect);
   printf("yCorrect= %f\n",yCorrect);
   fclose(fp);

}
//------------------------------------------------------------------------------------------------
// END  static void readOffsets
//------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------
// static bool compare_float
//------------------------------------------------------------------------------------------------
static bool compare_float(float x, float y, float epsilon)
{
   if(fabs(x - y) < epsilon)
      return true; //they are same
      return false; //they are not same
}
//------------------------------------------------------------------------------------------------
// END static bool compare_float
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void ServoAngle
//------------------------------------------------------------------------------------------------
static void ServoAngle(int Num,float &Angle)     
{
  pthread_mutex_lock(&I2C_Mutex);
  if (Num==TILT_SERVO)
   {
     if (Angle< MIN_TILT) Angle=MIN_TILT; 
     else if (Angle > MAX_TILT) Angle=MAX_TILT; 
   }
  else if (Num==PAN_SERVO)
   {
    if (Angle< MIN_PAN) Angle = MIN_PAN;
    else if (Angle > MAX_PAN) Angle=MAX_PAN;
   }
  Servos->angle(Num,Angle);
  pthread_mutex_unlock(&I2C_Mutex);
} 
//------------------------------------------------------------------------------------------------
// END static void ServoAngle
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void fire
//------------------------------------------------------------------------------------------------
static void fire(bool value)
{
 pthread_mutex_lock(&GPIO_Mutex);
 if (value) SystemState=(SystemState_t)(SystemState|FIRING);
 else SystemState=(SystemState_t)(SystemState & CLEAR_FIRING_MASK);
 gpioWrite(29,value);
 pthread_mutex_unlock(&GPIO_Mutex);
}
//------------------------------------------------------------------------------------------------
// END static void fire
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void armed
//------------------------------------------------------------------------------------------------
static void armed(bool value)
{
  pthread_mutex_lock(&GPIO_Mutex);
  if (value) SystemState=(SystemState_t)(SystemState | ARMED);
  else SystemState=(SystemState_t)(SystemState & CLEAR_ARMED_MASK);
  pthread_mutex_unlock(&GPIO_Mutex);
}
//------------------------------------------------------------------------------------------------
// END static void armed
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void calibrate
//------------------------------------------------------------------------------------------------
static void calibrate(bool value)
{
  pthread_mutex_lock(&GPIO_Mutex);
  if (value) SystemState=(SystemState_t)(SystemState|CALIB_ON);
  else SystemState=(SystemState_t)(SystemState & CLEAR_CALIB_MASK);
  pthread_mutex_unlock(&GPIO_Mutex);
}
//------------------------------------------------------------------------------------------------
// END static void calibrate
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void laser
//------------------------------------------------------------------------------------------------
static void laser(bool value)
{
  pthread_mutex_lock(&GPIO_Mutex);
  if (value) SystemState=(SystemState_t)(SystemState|LASER_ON);
  else SystemState=(SystemState_t)(SystemState & CLEAR_LASER_MASK);
  gpioWrite(31,value);
  pthread_mutex_unlock(&GPIO_Mutex);
}
//------------------------------------------------------------------------------------------------
// END static void laser
//------------------------------------------------------------------------------------------------

// 발사 제어를 위한 새로운 스레드
void* fire_control_thread(void* arg) {
    while(true) {
        if(fireControl.is_firing) {
            auto current_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - fireControl.last_fire_time).count();
            
            if(duration >= FireControl::FIRE_DURATION_MS) {
                fire(false);
                fireControl.is_firing = false;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return nullptr;
}


//------------------------------------------------------------------------------------------------
// static void ProcessTargetEngagements
//------------------------------------------------------------------------------------------------
// Shiwon
static void ProcessTargetEngagements(TAutoEngage *Auto,int width,int height, const std::vector<BoundingBox>& bboxes)
{
 
 bool NewState=false;
        
 switch(Auto->State)
  {
   case NOT_ACTIVE:
                   break;
   case ACTIVATE:
                   Auto->CurrentIndex=0;
                   Auto->State=NEW_TARGET;

   case NEW_TARGET:
                   AutoEngage.Target=0;
                   Auto->StableCount=0;
                   Auto->LastPan=-99999.99;
                   Auto->LastTilt=-99999.99;
                   NewState=true;

  case LOOKING_FOR_TARGET:
  case TRACKING:
    {
      // printBoundingBoxes(bboxes);

      TEngagementState state=LOOKING_FOR_TARGET;

      // Declare vectors for bad and good bounding boxes
      std::vector<BoundingBox> bad_bboxes;
      std::vector<BoundingBox> good_bboxes;

      // Separate bboxes into bad_bboxes and good_bboxes
      for (const auto& res : bboxes) {
          if (res.class_id != 2) {
          // if (res.class_id != 1) {
              bad_bboxes.push_back(res);
          } else if (res.class_id == 2) {
          // } else if (res.class_id == 1) {
              good_bboxes.push_back(res);
          }
      }

      // Check if any bad bbox center is inside any good bbox
      for (const auto& good_bbox : good_bboxes) {
        bool overlapped = false;
        for (const auto& bad_bbox : bad_bboxes) {
            float bad_center_x = bad_bbox.center_x;
            float bad_center_y = bad_bbox.center_y;

            if (bad_center_x >= good_bbox.x_min && bad_center_x <= good_bbox.x_max &&
                bad_center_y >= good_bbox.y_min && bad_center_y <= good_bbox.y_max) {
                // The bad bbox center is inside the good bbox
                std::cout << "Bad bbox center (" << bad_center_x << ", " << bad_center_y
                          << ") is inside good bbox with corners ("
                          << good_bbox.x_min << ", " << good_bbox.y_min << ") to ("
                          << good_bbox.x_max << ", " << good_bbox.y_max << ")\n";
                overlapped = true; // Mark as overlapped
                break; // No need to check further bad bboxes

            }
        }

      // Track good target
        float PanError,TiltError;
        float center_x = good_bbox.center_x;
        float center_y = good_bbox.center_y;

        //xCorrect = xCorrect / 3;
        //yCorrect = yCorrect / 2.25;

        PanError=(center_x+xCorrect)-width/2;
        TiltError=(center_y+yCorrect)-height/2;

        #if STATIONARY_SHOOTING == 1 // stable target
            #define SMOOTH_FACTOR 70.0f
        #else // moving target
            #define SMOOTH_FACTOR 20.0f
        #endif

        if (fabs(PanError) > 0.75f) {
            Pan = Pan - PanError / SMOOTH_FACTOR;
            // Pan = Pan-PanError/(20/speed_factor);
            ServoAngle(PAN_SERVO, Pan);
        }

        if (fabs(TiltError) > 1.5f) {
            Tilt = Tilt - TiltError / SMOOTH_FACTOR;
            // Tilt = Tilt-TiltError/(20/speed_factor);
            ServoAngle(TILT_SERVO, Tilt);
        }

        if (fabs(Pan - Auto->LastPan) > 0.75f) {
            ServoAngle(PAN_SERVO, Pan);
        }
        if (fabs(Tilt - Auto->LastTilt) > 1.5f) {
            ServoAngle(TILT_SERVO, Tilt);
        }        

        if ((compare_float(Auto->LastPan,Pan)) && (compare_float(Auto->LastTilt,Tilt)))
        {
          Auto->StableCount++;
        }
        else Auto->StableCount = std::max(0, Auto->StableCount - 2);

        Auto->LastPan=Pan;
        Auto->LastTilt=Tilt;

        const float thres = 0.5;

        #if STATIONARY_SHOOTING == 1
        usleep(100000);
        // ✅ Tracking 중에는 절대 fire 하지 않음
        fire(false);
        #endif

        // If confidence score exceeds threshold and no bad bboxes overlap with the good target, perform shoot
        if (good_bbox.conf>thres and !overlapped) {
            std::cout << "Good bbox with corners ("
                      << good_bbox.x_min << ", " << good_bbox.y_min << ") to ("
                      << good_bbox.x_max << ", " << good_bbox.y_max << ") is clear and stable (conf:" << good_bbox.conf << "). Shooting!\n";
            if(!fireControl.is_firing) {
                fire(true);
                fireControl.is_firing = true;
                fireControl.last_fire_time = std::chrono::steady_clock::now();
                printf("PanError: %lf, TiltError: %lf\n", PanError, TiltError);
            }
            // break;
        } else {
            if (overlapped) {
              std::cout << "Good bbox with corners ("
                        << good_bbox.x_min << ", " << good_bbox.y_min << ") to ("
                        << good_bbox.x_max << ", " << good_bbox.y_max << ") overlaps with a bad bbox (conf:" << good_bbox.conf << "). Skipping.\n";
            }
            else if (Auto->StableCount<=2) {
              std::cout << "Good bbox with corners ("
                        << good_bbox.x_min << ", " << good_bbox.y_min << ") to ("
                        << good_bbox.x_max << ", " << good_bbox.y_max << ") is unstable (conf:" << good_bbox.conf << "). Skipping.\n";

            }
            state=TRACKING;
        }
        break;
    }
  }
  break;
   case ENGAGEMENT_IN_PROGRESS:
                {
                  EngagementCustom();
                  Auto->State=ENGAGEMENT_COMPLETE;
                }
                break;      
   case ENGAGEMENT_COMPLETE:
                {
                 printf("Engagment Complete!\n");
                 Auto->StableCount=0;
                 Auto->LastPan=-99999.99;
                 Auto->LastTilt=-99999.99;
                 Auto->State=TRACKING;
                }
                break;  
    case TRACKING_STABLE:
                break;
    default: 
             printf("Invaid State\n");
             break;    
 }
  return;
}
//------------------------------------------------------------------------------------------------
// static void ProcessTargetEngagements
//------------------------------------------------------------------------------------------------
// Soomin: 하나의 Target 쏘는 동안 다른 target tracking 안하는 코드를 의도했으나 잘 작동안하는ㄱ듯
// static void ProcessTargetEngagements(TAutoEngage *Auto, int width, int height, const std::vector<BoundingBox>& bboxes)
// {
//     bool NewState = false;
        
//     switch(Auto->State)
//     {
//         case NOT_ACTIVE:
//             break;
            
//         case ACTIVATE:
//             Auto->CurrentIndex = 0;
//             Auto->State = NEW_TARGET;
//             // fall-through

//         case NEW_TARGET:
//             // 타겟 선택 초기화: STATIONARY_SHOOTING 모드에서는 아직 타겟이 정해지지 않았음을 -1로 표시
//             AutoEngage.Target = -1;
//             Auto->StableCount = 0;
//             Auto->LastPan = -99999.99;
//             Auto->LastTilt = -99999.99;
//             NewState = true;
//             // fall-through

//         case LOOKING_FOR_TARGET:
//         case TRACKING:
//         {
//             TEngagementState state = LOOKING_FOR_TARGET;

//             // 1. bad_bboxes (예: cat 등)와 good_bboxes (좀비)를 분리합니다.
//             std::vector<BoundingBox> bad_bboxes;
//             std::vector<BoundingBox> good_bboxes;
//             for (const auto& res : bboxes) {
//                 if (res.class_id != 2) {
//                     bad_bboxes.push_back(res);
//                 } else {  // res.class_id == 2 (좀비)
//                     good_bboxes.push_back(res);
//                 }
//             }
            
//             if(good_bboxes.empty())
//                 break;  // good target이 하나도 없으면 아무것도 하지 않음

//             // STATIONARY_SHOOTING 모드에 따라 타겟 선택 및 처리 방식을 달리합니다.
//             #if STATIONARY_SHOOTING == 1
//                 {
//                     // (1) 타겟이 아직 선택되지 않았다면(good target 리스트 중 가장 큰 객체를 선택)
//                     if (Auto->Target < 0) {
//                         // 크기(면적) 기준 내림차순 정렬: 면적이 큰(즉, 가까워 보이는) 좀비가 우선
//                         std::sort(good_bboxes.begin(), good_bboxes.end(),
//                             [](const BoundingBox& a, const BoundingBox& b) {
//                                 float area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min);
//                                 float area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min);
//                                 return area_a > area_b;
//                             });
//                         Auto->Target = 0;  // 가장 큰 타겟을 선택하고 "락"함.
//                         std::cout << "Locked on new target (STATIONARY_SHOOTING mode)" << std::endl;
//                     }
//                     // (2) 선택된 타겟(인덱스 Auto->Target)만 처리합니다.
//                     // 만약 리스트 크기가 변경되어 인덱스가 범위를 벗어나면 재설정
//                     if (Auto->Target >= (int)good_bboxes.size())
//                         Auto->Target = 0;
//                     const BoundingBox &target_bbox = good_bboxes[Auto->Target];
                    
//                     // bad_bboxes와의 겹침 검사 (타겟 bbox 내에 bad 객체가 있는지 확인)
//                     bool overlapped = false;
//                     for (const auto& bad_bbox : bad_bboxes) {
//                         if (bad_bbox.center_x >= target_bbox.x_min && bad_bbox.center_x <= target_bbox.x_max &&
//                             bad_bbox.center_y >= target_bbox.y_min && bad_bbox.center_y <= target_bbox.y_max)
//                         {
//                             std::cout << "Bad bbox center (" << bad_bbox.center_x << ", " << bad_bbox.center_y
//                                       << ") is inside target bbox with corners ("
//                                       << target_bbox.x_min << ", " << target_bbox.y_min << ") to ("
//                                       << target_bbox.x_max << ", " << target_bbox.y_max << ")\n";
//                             overlapped = true;
//                             break;
//                         }
//                     }
                    
//                     // (3) 타겟 트래킹: 화면 중앙과 target_bbox의 center 차이 계산
//                     float PanError, TiltError;
//                     float center_x = target_bbox.center_x;
//                     float center_y = target_bbox.center_y;
//                     PanError = (center_x + xCorrect) - width / 2;
//                     TiltError = (center_y + yCorrect) - height / 2;
                    
//                     #if STATIONARY_SHOOTING == 1  // stable target: 좀 더 부드럽게 이동
//                         #define SMOOTH_FACTOR 70.0f
//                     #else  // moving target
//                         #define SMOOTH_FACTOR 20.0f
//                     #endif
                    
//                     if (fabs(PanError) > 0.75f) {
//                         Pan = Pan - PanError / SMOOTH_FACTOR;
//                         ServoAngle(PAN_SERVO, Pan);
//                     }
//                     if (fabs(TiltError) > 1.5f) {
//                         Tilt = Tilt - TiltError / SMOOTH_FACTOR;
//                         ServoAngle(TILT_SERVO, Tilt);
//                     }
//                     if (fabs(Pan - Auto->LastPan) > 0.75f) {
//                         ServoAngle(PAN_SERVO, Pan);
//                     }
//                     if (fabs(Tilt - Auto->LastTilt) > 1.5f) {
//                         ServoAngle(TILT_SERVO, Tilt);
//                     }
                    
//                     if (compare_float(Auto->LastPan, Pan) && compare_float(Auto->LastTilt, Tilt))
//                         Auto->StableCount++;
//                     else
//                         Auto->StableCount = std::max(0, Auto->StableCount - 2);
                    
//                     Auto->LastPan = Pan;
//                     Auto->LastTilt = Tilt;
                    
//                     const float thres = 0.5f;
                    
//                     // (4) 사격 조건: confidence가 임계치 이상, 타겟이 안정적이며, bad 객체와 겹치지 않을 때
//                     if (target_bbox.conf > thres && !overlapped && Auto->StableCount > 2) {
//                         std::cout << "Target bbox ("
//                                   << target_bbox.x_min << ", " << target_bbox.y_min << ") to ("
//                                   << target_bbox.x_max << ", " << target_bbox.y_max << ") is clear and stable (conf:" 
//                                   << target_bbox.conf << "). Shooting!\n";
//                         if (!fireControl.is_firing) {
//                             fire(true);
//                             fireControl.is_firing = true;
//                             fireControl.last_fire_time = std::chrono::steady_clock::now();
//                             printf("PanError: %lf, TiltError: %lf\n", PanError, TiltError);
//                         }
//                         // 사격이 시작되었으므로, 이 타겟에 대한 처리를 끝낸 후에는 타겟 "락"을 해제하지 않습니다.
//                         // (즉, ENGAGEMENT 상태에서 완료될 때까지 계속 이 타겟을 추적)
//                     }
//                     else {
//                         if (overlapped) {
//                             std::cout << "Target bbox ("
//                                       << target_bbox.x_min << ", " << target_bbox.y_min << ") to ("
//                                       << target_bbox.x_max << ", " << target_bbox.y_max << ") overlaps with a bad bbox (conf:" 
//                                       << target_bbox.conf << "). Skipping.\n";
//                         }
//                         else if (Auto->StableCount <= 2) {
//                             std::cout << "Target bbox ("
//                                       << target_bbox.x_min << ", " << target_bbox.y_min << ") to ("
//                                       << target_bbox.x_max << ", " << target_bbox.y_max << ") is unstable (conf:" 
//                                       << target_bbox.conf << "). Skipping.\n";
//                         }
//                         state = TRACKING;
//                     }
//                 }
//             #else  // STATIONARY_SHOOTING == 0: 기존 로직(매 프레임 첫 good_bbox 처리)
//                 {
//                     for (const auto& good_bbox : good_bboxes) {
//                         bool overlapped = false;
//                         for (const auto& bad_bbox : bad_bboxes) {
//                             float bad_center_x = bad_bbox.center_x;
//                             float bad_center_y = bad_bbox.center_y;
//                             if (bad_center_x >= good_bbox.x_min && bad_center_x <= good_bbox.x_max &&
//                                 bad_center_y >= good_bbox.y_min && bad_center_y <= good_bbox.y_max) {
//                                 std::cout << "Bad bbox center (" << bad_center_x << ", " << bad_center_y
//                                           << ") is inside good bbox with corners ("
//                                           << good_bbox.x_min << ", " << good_bbox.y_min << ") to ("
//                                           << good_bbox.x_max << ", " << good_bbox.y_max << ")\n";
//                                 overlapped = true;
//                                 break;
//                             }
//                         }
//                         float PanError, TiltError;
//                         float center_x = good_bbox.center_x;
//                         float center_y = good_bbox.center_y;
//                         PanError = (center_x + xCorrect) - width / 2;
//                         TiltError = (center_y + yCorrect) - height / 2;
                        
//                         #if STATIONARY_SHOOTING == 1
//                             #define SMOOTH_FACTOR 70.0f
//                         #else
//                             #define SMOOTH_FACTOR 20.0f
//                         #endif
                        
//                         if (fabs(PanError) > 0.75f) {
//                             Pan = Pan - PanError / SMOOTH_FACTOR;
//                             ServoAngle(PAN_SERVO, Pan);
//                         }
//                         if (fabs(TiltError) > 1.5f) {
//                             Tilt = Tilt - TiltError / SMOOTH_FACTOR;
//                             ServoAngle(TILT_SERVO, Tilt);
//                         }
//                         if (fabs(Pan - Auto->LastPan) > 0.75f) {
//                             ServoAngle(PAN_SERVO, Pan);
//                         }
//                         if (fabs(Tilt - Auto->LastTilt) > 1.5f) {
//                             ServoAngle(TILT_SERVO, Tilt);
//                         }
//                         if (compare_float(Auto->LastPan, Pan) && compare_float(Auto->LastTilt, Tilt))
//                             Auto->StableCount++;
//                         else
//                             Auto->StableCount = std::max(0, Auto->StableCount - 2);
                        
//                         Auto->LastPan = Pan;
//                         Auto->LastTilt = Tilt;
                        
//                         const float thres = 0.5f;
//                         if (good_bbox.conf > thres && !overlapped) {
//                             std::cout << "Good bbox with corners ("
//                                       << good_bbox.x_min << ", " << good_bbox.y_min << ") to ("
//                                       << good_bbox.x_max << ", " << good_bbox.y_max << ") is clear and stable (conf:" 
//                                       << good_bbox.conf << "). Shooting!\n";
//                             if (!fireControl.is_firing) {
//                                 fire(true);
//                                 fireControl.is_firing = true;
//                                 fireControl.last_fire_time = std::chrono::steady_clock::now();
//                                 printf("PanError: %lf, TiltError: %lf\n", PanError, TiltError);
//                             }
//                         }
//                         else {
//                             if (overlapped) {
//                                 std::cout << "Good bbox with corners ("
//                                           << good_bbox.x_min << ", " << good_bbox.y_min << ") to ("
//                                           << good_bbox.x_max << ", " << good_bbox.y_max << ") overlaps with a bad bbox (conf:" 
//                                           << good_bbox.conf << "). Skipping.\n";
//                             }
//                             else if (Auto->StableCount <= 2) {
//                                 std::cout << "Good bbox with corners ("
//                                           << good_bbox.x_min << ", " << good_bbox.y_min << ") to ("
//                                           << good_bbox.x_max << ", " << good_bbox.y_max << ") is unstable (conf:" 
//                                           << good_bbox.conf << "). Skipping.\n";
//                             }
//                             state = TRACKING;
//                         }
//                         break;  // 한 타겟만 처리하고 빠져나옴.
//                     }
//                 }
//             #endif // STATIONARY_SHOOTING
            
//             break;
//         }
        
//         case ENGAGEMENT_IN_PROGRESS:
//         {
//             // ENGAGEMENT 모드에서는 Shooting/EngagementCustom() 호출 후에 타겟 "락"을 해제하도록 처리
//             EngagementCustom();
//             // 사격이 완료되었으므로, 다음 타겟 추적을 위해 타겟 락을 해제(예: -1로 초기화)
//             #if STATIONARY_SHOOTING == 1
//                 Auto->Target = -1;
//             #endif
//             Auto->State = ENGAGEMENT_COMPLETE;
//             break;
//         }
        
//         case ENGAGEMENT_COMPLETE:
//         {
//             printf("Engagement Complete!\n");
//             Auto->StableCount = 0;
//             Auto->LastPan = -99999.99;
//             Auto->LastTilt = -99999.99;
//             Auto->State = TRACKING;
//             break;
//         }
        
//         case TRACKING_STABLE:
//             break;
            
//         default:
//             printf("Invalid State\n");
//             break;
//     }
//     return;
// }
//------------------------------------------------------------------------------------------------
// END static void ProcessTargetEngagements
//------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------
// END static void ProcessTargetEngagements
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void CreateNoDataAvalable
//------------------------------------------------------------------------------------------------
static void CreateNoDataAvalable(void)
{
  while (!GetFrame(NoDataAvalable)) printf("blank frame grabbed\n");    
  cv::String Text =format("NO DATA");

  int baseline;
  float FontSize=3.0; //12.0;
  int Thinkness=4;
    
  NoDataAvalable.setTo(cv::Scalar(128, 128, 128));
  Size TextSize= cv::getTextSize(Text, cv::FONT_HERSHEY_COMPLEX, FontSize,  Thinkness,&baseline); // Get font size

  int textX = (NoDataAvalable.cols- TextSize.width) / 2;
  int textY = (NoDataAvalable.rows + TextSize.height) / 2;
  putText(NoDataAvalable,Text,Point(textX , textY),cv::FONT_HERSHEY_COMPLEX,FontSize,Scalar(255,255,255),Thinkness*Thinkness,cv::LINE_AA);
  putText(NoDataAvalable,Text,Point(textX , textY),cv::FONT_HERSHEY_COMPLEX,FontSize,Scalar(0,0,0),Thinkness,cv::LINE_AA);
  printf("frame size %d %d\n", NoDataAvalable.cols,NoDataAvalable.rows);
}
//------------------------------------------------------------------------------------------------
// END static void CreateNoDataAvalable
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static bool OpenCamera
//------------------------------------------------------------------------------------------------
static bool OpenCamera(void)
{
#if 0
    std::string pipeline = gstreamer_pipeline(capture_width,
  capture_height,
  display_width,
  display_height,
  framerate,
  flip_method);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";
    capture=new cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
#else
    capture=new cv::VideoCapture("/dev/video0",cv::CAP_V4L);
    capture->set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    capture->set(cv::CAP_PROP_FRAME_HEIGHT,1200);
    capture->set(cv::CAP_PROP_FPS, 30);
#endif
    if(!capture->isOpened()) {
        std::cout<<"Failed to open camera."<<std::endl;
        delete capture;
        return false;
    }

 return(true);
}
//------------------------------------------------------------------------------------------------
// END static bool OpenCamera
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static bool GetFrame
//------------------------------------------------------------------------------------------------
static bool GetFrame(Mat &frame)
{
    // wait for a new frame from camera and store it into 'frame'
    capture->read(frame);
    // check if we succeeded
    if (frame.empty()) return(false);

    flip(frame, frame,-1);       // if running on PI5 flip(-1)=180 degrees

    return (true);
}
//------------------------------------------------------------------------------------------------
// END static bool GetFrame
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void CloseCamera
//------------------------------------------------------------------------------------------------
static void CloseCamera(void)
{
 if (capture!=NULL)  
 {
       capture->release();
       delete capture;
       capture=NULL;
 }
}
//------------------------------------------------------------------------------------------------
// END static void CloseCamera
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void OpenServos
//------------------------------------------------------------------------------------------------
static void OpenServos(void)
{
 Servos = new Servo(0x40, 0.750, 2.250);
}
//------------------------------------------------------------------------------------------------
// END static void OpenServos
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static bool CloseServos
//------------------------------------------------------------------------------------------------
static void CloseServos(void)
{
 if (Servos!=NULL)
  {
   delete Servos;
   Servos=NULL;
  }
}
//------------------------------------------------------------------------------------------------
// END static  CloseServos
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void OpenGPIO
//------------------------------------------------------------------------------------------------
static void OpenGPIO(void)
{
  int Init;

 Init = gpioInitialise();
 if (Init < 0)
    {
      /* jetgpio initialisation failed */
      printf("Jetgpio initialisation failed. Error code:  %d\n", Init);
      exit(Init);
    }
  else
    {
      /* jetgpio initialised okay*/
      printf("Jetgpio initialisation OK. Return code:  %d\n", Init);
    } 

  // Setting up pin 38 as OUTPUT

  gpioSetMode(29, JET_OUTPUT); // Fire Cannon
  gpioSetMode(31, JET_OUTPUT); // Laser

}
//------------------------------------------------------------------------------------------------
// END static void OpenGPIO
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void CloseGPIO
//------------------------------------------------------------------------------------------------
static void CloseGPIO(void)
{
 gpioTerminate();
}
//------------------------------------------------------------------------------------------------
// END static void CloseGPIO
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static bool OLEDInit
//------------------------------------------------------------------------------------------------
static bool OLEDInit(void)
{
    uint8_t rc = 0;
    // open the I2C device node
    rc = ssd1306_init(i2c_node_address);
    
    if (rc != 0)
    {
        printf("no oled attached to /dev/i2c-%d\n", i2c_node_address);
        return (false);
    }
   rc= ssd1306_oled_default_config(64, 128);
    if (rc != 0)
    {
        printf("OLED DIsplay initialization failed\n");
        return (false);
    }
    rc=ssd1306_oled_clear_screen();
    if (rc != 0)
    {
        printf("OLED Clear screen Failed\n");
        return (false);

    }
  ssd1306_oled_set_rotate(0);
  ssd1306_oled_set_XY(0, 0);
  ssd1306_oled_write_line(OLED_Font, (char *) "READY");
  return(true); 
}
//------------------------------------------------------------------------------------------------
// END static bool OLEDInit
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void OLED_UpdateStatus
//------------------------------------------------------------------------------------------------
static void OLED_UpdateStatus(void)
{
    char Status[128];
    static SystemState_t LastSystemState=UNKNOWN;
    static SystemState_t LastSystemStateBase=UNKNOWN;
    SystemState_t SystemStateBase;
    if (!HaveOLED) return;
    pthread_mutex_lock(&I2C_Mutex);
    if (LastSystemState==SystemState)
       {
        pthread_mutex_unlock(&I2C_Mutex);
        return;
       }
    SystemStateBase=(SystemState_t)(SystemState & CLEAR_LASER_FIRING_ARMED_CALIB_MASK);
    if (SystemStateBase!=LastSystemStateBase)
      {
       LastSystemStateBase=SystemStateBase;
       ssd1306_oled_clear_line(0);  
       ssd1306_oled_set_XY(0, 0);
       if  (SystemStateBase==UNKNOWN)  strcpy(Status,"Unknown");
       else if  (SystemStateBase==SAFE)  strcpy(Status,"SAFE");
       else if  (SystemStateBase==PREARMED)  strcpy(Status,"PREARMED");
       else if  (SystemStateBase==ENGAGE_AUTO)  strcpy(Status,"ENGAGE AUTO");
       else if  (SystemStateBase==ARMED_MANUAL)  strcpy(Status,"ARMED_MANUAL");
       if (SystemState & ARMED) strcat(Status,"-ARMED");
       ssd1306_oled_write_line(OLED_Font, Status);
      }

   if((SystemState & LASER_ON)!=(LastSystemState & LASER_ON)||(LastSystemState==UNKNOWN))
    {
     ssd1306_oled_clear_line(1); 
     ssd1306_oled_set_XY(0, 1);
     if (SystemState & LASER_ON ) strcpy(Status,"LASER-ON");
     else strcpy(Status,"LASER-OFF");
     ssd1306_oled_write_line(OLED_Font, Status);
    }
   if((SystemState & FIRING)!=(LastSystemState & FIRING)||(LastSystemState==UNKNOWN))
   {
     ssd1306_oled_clear_line(2); 
     ssd1306_oled_set_XY(0, 2);
     if (SystemState & FIRING ) strcpy(Status,"FIRING-TRUE");
     else strcpy(Status,"FIRING-FALSE");
     ssd1306_oled_write_line(OLED_Font, Status);
    }
   LastSystemState=SystemState;
   pthread_mutex_unlock(&I2C_Mutex);
   return;
}
//------------------------------------------------------------------------------------------------
// END static void OLED_UpdateStatus
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void DrawCrosshair
//------------------------------------------------------------------------------------------------
static void DrawCrosshair(Mat &img, Point correct, const Scalar &color)
{
  // Use `shift` to try to gain sub-pixel accuracy
  int shift = 10;
  int m = pow(2, shift);

  Point pt = Point((int)((img.cols/2-correct.x/2) * m), (int)((img.rows/2-correct.y/2) * m));

  int size = int(10 * m);
  int gap = int(4 * m);
  line(img, Point(pt.x, pt.y-size), Point(pt.x, pt.y-gap), color, 1,LINE_8, shift);
  line(img, Point(pt.x, pt.y+gap), Point(pt.x, pt.y+size), color, 1,LINE_8, shift);
  line(img, Point(pt.x-size, pt.y), Point(pt.x-gap, pt.y), color, 1,LINE_8, shift);
  line(img, Point(pt.x+gap, pt.y), Point(pt.x+size, pt.y), color, 1,LINE_8, shift);
  line(img, pt, pt, color, 1,LINE_8, shift);
}
//------------------------------------------------------------------------------------------------
// END static void DrawCrosshair
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// main - This is the main program for the Gel Cannon and contains the control loop
//------------------------------------------------------------------------------------------------

void printBoundingBoxes(const std::vector<BoundingBox>& bboxes) {
    for (size_t i = 0; i < bboxes.size(); ++i) {
        const BoundingBox& bbox = bboxes[i];
        printf("BoundingBox %zu: x1: %.2f, y1: %.2f, x2: %.2f, y2: %.2f, confidence: %.2f, class_id: %.0f\n",
               i + 1, bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max, bbox.conf, bbox.class_id);
    }
}

int main(int argc, const char** argv)
{
  Mat                              Frame,ResizedFrame;      // camera image in Mat format 
  float                            avfps=0.0,FPS[16]={0.0,0.0,0.0,0.0,
                                                      0.0,0.0,0.0,0.0,
                                                      0.0,0.0,0.0,0.0,
                                                      0.0,0.0,0.0,0.0};
  int                              retval,i,Fcnt = 0;
  struct sockaddr_in               cli_addr;
  socklen_t                        clilen;
  chrono::steady_clock::time_point Tbegin, Tend;
 
  ReadOffsets();

  for (i = 0; i < 16; i++) FPS[i] = 0.0;
    
  AutoEngage.State=TRACKING;
  AutoEngage.HaveFiringOrder=false;
  AutoEngage.NumberOfTartgets=0;

  pthread_mutexattr_init(&TCP_MutexAttr);
  pthread_mutexattr_settype(&TCP_MutexAttr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutexattr_init(&GPIO_MutexAttr);
  pthread_mutexattr_settype(&GPIO_MutexAttr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutexattr_init(&I2C_MutexAttr);
  pthread_mutexattr_settype(&I2C_MutexAttr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutexattr_init(&Engmnt_MutexAttr);
  pthread_mutexattr_settype(&Engmnt_MutexAttr, PTHREAD_MUTEX_ERRORCHECK);

  if (pthread_mutex_init(&TCP_Mutex, &TCP_MutexAttr)!=0) return -1;
  if (pthread_mutex_init(&GPIO_Mutex, &GPIO_MutexAttr)!=0) return -1; 
  if (pthread_mutex_init(&I2C_Mutex, &I2C_MutexAttr)!=0) return -1; 
  if (pthread_mutex_init(&Engmnt_Mutex, &Engmnt_MutexAttr)!=0) return -1; 

  // fire control 스레드 생성 추가
  pthread_t fire_thread;
  if (pthread_create(&fire_thread, nullptr, fire_control_thread, nullptr) != 0) {
    printf("Fire control thread creation failed\n");
    return -1;
  }  

  HaveOLED=OLEDInit();   

  printf("OpenCV: Version %s\n",cv::getVersionString().c_str());

  //printf("OpenCV: %s", cv::getBuildInformation().c_str());
  
#if USE_IMAGE_MATCH
  int shm_fd = shm_open(SHARED_MEMORY_NAME, O_RDONLY, 0666);
  if (shm_fd == -1) {
      std::cerr << "Failed to open shared memory" << std::endl;
      return 1;
  }

  void* map = mmap(0, MEMORY_SIZE, PROT_READ, MAP_SHARED, shm_fd, 0);
  if (map == MAP_FAILED) {
      std::cerr << "Failed to map shared memory" << std::endl;
      return 1;
  }

  sem_t* semaphore = sem_open(SHARED_MEMORY_NAME, 0);
  if (semaphore == SEM_FAILED) {
      std::cerr << "Failed to open semaphore" << std::endl;
      return 1;
  }
  printf("Image Match Mode\n");

  if (LoadRefImages(symbols) == -1) 
    {
      printf("Error reading reference symbols\n");
      return -1;
    }

#endif

 if  ((TcpListenPort=OpenTcpListenPort(PORT))==NULL)  // Open UDP Network port
     {
       printf("OpenTcpListenPortFailed\n");
       return(-1);
     }

   OpenGPIO();

   laser(false);
   fire(false);
   calibrate(false);
   
   OpenServos();
   ServoAngle(PAN_SERVO, Pan);
   ServoAngle(TILT_SERVO, Tilt);

   Setup_Control_C_Signal_Handler_And_Keyboard_No_Enter(); // Set Control-c handler to properly exit clean

  while(true) {
    #if USE_IMAGE_MATCH
        TEngagementState tmpstate=AutoEngage.State;
        // sem_wait(semaphore);
        int num_boxes;
        std::memcpy(&num_boxes, map, sizeof(int));
        
        std::vector<BoundingBox> bboxes(num_boxes);
        std::memcpy(bboxes.data(), (char*)map + sizeof(int), num_boxes * sizeof(BoundingBox));
        
        // printBoundingBoxes(bboxes);

        // printf("State: %d\n", AutoEngage.State);
        // printf("Center X: %lf\n", lastDetectedTargets[0].center.x);

        // if (tmpstate!=ENGAGEMENT_IN_PROGRESS) FindTargetsCustom(bboxes);
        ProcessTargetEngagements(&AutoEngage,640,480,bboxes);
        // ProcessTargetEngagements(&AutoEngage,1920,1080,bboxes);
        // if (tmpstate!=ENGAGEMENT_IN_PROGRESS) DrawTargets(Frame);
        // fire(false);
        sem_post(semaphore);
        HandleInputChar();
        //usleep(100000);
        usleep(60000);
    #else
        HandleInputChar();
    #endif
  }

  #if USE_IMAGE_MATCH
    munmap(map, MEMORY_SIZE);
    shm_unlink(SHARED_MEMORY_NAME);
    sem_close(semaphore);
  #endif
  
  printf("Main Thread Exiting\n");
  CleanUp();
  return 0;
}
//------------------------------------------------------------------------------------------------
// End main
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void * EngagementThread
//------------------------------------------------------------------------------------------------
static void * EngagementThread(void *data) 
{
  int ret;
  while (1) {
    if ((ret = pthread_mutex_lock(&Engmnt_Mutex)) != 0) {
      
      printf("Engmnt_Mutex ERROR\n");
      break;
    }
    printf("Waiting for Engagement Order\n");
    if ((ret = pthread_cond_wait(&Engagement_cv, &Engmnt_Mutex)) != 0) {
       printf("Engagement  pthread_cond_wait ERROR\n");
      break;

    }

    printf("Engagment in Progress\n");
    laser(true);
    SendSystemState(SystemState);
    usleep(1500*1000);
    fire(true);
    SendSystemState(SystemState);
    usleep(1500*1000);
    fire(false);
    laser(false);
    armed(false);
    SendSystemState(SystemState);
    PrintfSend("Engaged Target %d",AutoEngage.Target);
    AutoEngage.State=ENGAGEMENT_COMPLETE;

    if ((ret = pthread_mutex_unlock(&Engmnt_Mutex)) != 0) 
    {
        printf("Engagement pthread_cond_wait ERROR\n");
       break;
    }
  }

  return NULL;
}
//------------------------------------------------------------------------------------------------
// END static void * EngagementThread
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static int PrintfSend
//------------------------------------------------------------------------------------------------
static int PrintfSend(const char *fmt, ...) 
{
    char Buffer[2048];
    int  BytesWritten;
    int  retval;
    pthread_mutex_lock(&TCP_Mutex); 
    va_list args;
    va_start(args, fmt);
    BytesWritten=vsprintf(Buffer,fmt, args);
    va_end(args);
    if (BytesWritten>0)
      {
       TMesssageHeader MsgHdr;
       BytesWritten++;
       MsgHdr.Len=htonl(BytesWritten);
       MsgHdr.Type=htonl(MT_TEXT);
       if (WriteDataTcp(TcpConnectedPort,(unsigned char *)&MsgHdr, sizeof(TMesssageHeader))!=sizeof(TMesssageHeader)) 
           {
            pthread_mutex_unlock(&TCP_Mutex);
            return (-1);
           }
       retval=WriteDataTcp(TcpConnectedPort,(unsigned char *)Buffer,BytesWritten);
       pthread_mutex_unlock(&TCP_Mutex);
       return(retval);
      }
    else 
     {
      pthread_mutex_unlock(&TCP_Mutex);
      return(BytesWritten);
     }
}
//------------------------------------------------------------------------------------------------
// END static int PrintfSend
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static int SendSystemState
//------------------------------------------------------------------------------------------------
static int SendSystemState(SystemState_t State)
{
 TMesssageSystemState StateMsg;
 int                  retval;
 pthread_mutex_lock(&TCP_Mutex);
 StateMsg.State=(SystemState_t)htonl(State);
 StateMsg.Hdr.Len=htonl(sizeof(StateMsg.State));
 StateMsg.Hdr.Type=htonl(MT_STATE);
 OLED_UpdateStatus();
 retval=WriteDataTcp(TcpConnectedPort,(unsigned char *)&StateMsg,sizeof(TMesssageSystemState));
 pthread_mutex_unlock(&TCP_Mutex);
 return(retval);
} 
//------------------------------------------------------------------------------------------------
// END static int SendSystemState
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void ProcessPreArm
//------------------------------------------------------------------------------------------------
static void ProcessPreArm(char * Code)
{
 char Decode[]={0x61,0x60,0x76,0x75,0x67,0x7b,0x72,0x7c};

 if (SystemState==SAFE)
  {
    if ((Code[sizeof(Decode)]==0) && (strlen(Code)==sizeof(Decode)))
      { 
        for (int i=0;i<sizeof(Decode);i++) Code[i]^=Decode[i];
        if (strcmp((const char*)Code,"PREARMED")==0)
          {
            SystemState=PREARMED;
            SendSystemState(SystemState);
          } 
      }
  }
}
//------------------------------------------------------------------------------------------------
// END static void ProcessPreArm
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void ProcessStateChangeRequest
//------------------------------------------------------------------------------------------------
static void ProcessStateChangeRequest(SystemState_t state)
{  
 static bool CalibrateWasOn=false;
 switch(state&CLEAR_LASER_FIRING_ARMED_CALIB_MASK)
 {
  case SAFE:
            {
              laser(false);
              calibrate(false);
              fire(false);
              SystemState=(SystemState_t)(state & CLEAR_LASER_FIRING_ARMED_CALIB_MASK);
              AutoEngage.State=NOT_ACTIVE;
              AutoEngage.HaveFiringOrder=false;
              AutoEngage.NumberOfTartgets=0;
            }
            break;
  case PREARMED:
            { 
              if (((SystemState&CLEAR_LASER_FIRING_ARMED_CALIB_MASK)==ENGAGE_AUTO) || 
                  ((SystemState&CLEAR_LASER_FIRING_ARMED_CALIB_MASK)==ARMED_MANUAL))
                {
                  laser(false);
                  fire(false);
                  calibrate(false);
                  if ((SystemState&CLEAR_LASER_FIRING_ARMED_CALIB_MASK)==ENGAGE_AUTO)
                     {
                      AutoEngage.State=NOT_ACTIVE;
                      AutoEngage.HaveFiringOrder=false;
                      AutoEngage.NumberOfTartgets=0;
                     }
                  SystemState=(SystemState_t)(state & CLEAR_LASER_FIRING_ARMED_CALIB_MASK);
                }
             }
             break;

  case ENGAGE_AUTO:
            {
              if ((SystemState&CLEAR_LASER_FIRING_ARMED_CALIB_MASK)!=PREARMED)
              {
               PrintfSend("Invalid State request to Auto %d\n",SystemState); 
              }
             else if (!AutoEngage.HaveFiringOrder)
              {
               PrintfSend("No Firing Order List");
              }
             else 
              {
                laser(false);
                calibrate(false);
                fire(false);
                SystemState=(SystemState_t)(state & CLEAR_LASER_FIRING_ARMED_CALIB_MASK);
                AutoEngage.State=ACTIVATE;
              }
            }
            break;
  case ARMED_MANUAL:
            {
              if (((SystemState&CLEAR_LASER_FIRING_ARMED_CALIB_MASK)!=PREARMED) && 
                  ((SystemState&CLEAR_LASER_FIRING_ARMED_CALIB_MASK)!=ARMED_MANUAL)) 
              {
               PrintfSend("Invalid State request to Auto %d\n",SystemState); 
              }
             else if ((SystemState&CLEAR_LASER_FIRING_ARMED_CALIB_MASK)==PREARMED)
              {
                laser(false);
                calibrate(false);
                fire(false);
                SystemState=(SystemState_t)(state & CLEAR_LASER_FIRING_ARMED_CALIB_MASK);
              }
             else SystemState=state;

            }
            break;
  default:
             {
              printf("UNKNOWN STATE REQUEST %d\n",state);
             }
              break;

 }

 if (SystemState & LASER_ON)  laser(true);
 else laser(false);

 if (SystemState & CALIB_ON)  
    {
     calibrate(true);
     CalibrateWasOn=true;
    }
 else 
    {
     calibrate(false);
     if (CalibrateWasOn) 
        {
         CalibrateWasOn=false;
         WriteOffsets();
        }
    }

 SendSystemState(SystemState);
}
//------------------------------------------------------------------------------------------------
// END static void ProcessStateChangeRequest
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void ProcessFiringOrder
//------------------------------------------------------------------------------------------------
static void ProcessFiringOrder(char * FiringOrder)
{
  int len=strlen(FiringOrder);
  
  AutoEngage.State=NOT_ACTIVE;
  AutoEngage.HaveFiringOrder=false;
  AutoEngage.NumberOfTartgets=0;
  AutoEngage.Target=0;

  if (len>10) 
     {
      printf("Firing order error\n");
      return; 
     }
  for (int i=0;i<len;i++)
    {
      AutoEngage.FiringOrder[i]=FiringOrder[i]-'0';
    }
  if (len>0)  AutoEngage.HaveFiringOrder=true;
  else
    {
     AutoEngage.HaveFiringOrder=false;
     PrintfSend("Empty Firing List");
     return;
    }
  AutoEngage.NumberOfTartgets=len; 
#if 0  
  printf("Firing order:\n");
  for (int i=0;i<len;i++) printf("%d\n",AutoEngage.FiringOrder[i]);
  printf("\n\n");
#endif
}
//------------------------------------------------------------------------------------------------
// END static void ProcessFiringOrder
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void ProcessCommands
//------------------------------------------------------------------------------------------------
static void ProcessCommands(unsigned char cmd)
{
 if (((SystemState & CLEAR_LASER_FIRING_ARMED_CALIB_MASK)!=PREARMED) &&
     ((SystemState & CLEAR_LASER_FIRING_ARMED_CALIB_MASK)!=ARMED_MANUAL))
    {
      printf("received Commands outside of Pre-Arm or Armed Manual State %x \n",cmd);
      return;
    } 
 if (((cmd==FIRE_START) || (cmd==FIRE_STOP)) && ((SystemState & CLEAR_LASER_FIRING_ARMED_CALIB_MASK)!=ARMED_MANUAL))
    {
      printf("received Fire Commands outside of Armed Manual State %x \n",cmd);
      return;
    } 


      switch(cmd)
        {
         case PAN_LEFT_START:
              RunCmds|=PAN_LEFT_START;
              RunCmds&=PAN_RIGHT_STOP;
              Pan+=INC;
              ServoAngle(PAN_SERVO, Pan);
              break;
         case PAN_RIGHT_START:
              RunCmds|=PAN_RIGHT_START;
              RunCmds&=PAN_LEFT_STOP;
              Pan-=INC;
              ServoAngle(PAN_SERVO, Pan);
              break;
         case PAN_UP_START:
              RunCmds|=PAN_UP_START;
              RunCmds&=PAN_DOWN_STOP;
              Tilt+=INC; 
              ServoAngle(TILT_SERVO, Tilt);
              break;
         case PAN_DOWN_START:
              RunCmds|=PAN_DOWN_START;
              RunCmds&=PAN_UP_STOP;
              Tilt-=INC; 
              ServoAngle(TILT_SERVO, Tilt);
              break;
         case FIRE_START:
              RunCmds|=FIRE_START;
              fire(true);
              SendSystemState(SystemState);
              break;   
         case PAN_LEFT_STOP:
              RunCmds&=PAN_LEFT_STOP;
              break;
         case PAN_RIGHT_STOP:
              RunCmds&=PAN_RIGHT_STOP;
              break;
         case PAN_UP_STOP:
              RunCmds&=PAN_UP_STOP;
              break;
         case PAN_DOWN_STOP:
              RunCmds&=PAN_DOWN_STOP;
              break;
         case FIRE_STOP: 
              RunCmds&=FIRE_STOP;
              fire(false);
              SendSystemState(SystemState);
              break;
         default:
              printf("invalid command %x\n",cmd);
              break;
      }

}
//------------------------------------------------------------------------------------------------
// END static void ProcessCommands
//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------
// static void ProcessCalibCommands
//------------------------------------------------------------------------------------------------
static void ProcessCalibCommands(unsigned char cmd)
{
 if (((SystemState & CLEAR_LASER_FIRING_ARMED_CALIB_MASK)!=PREARMED) &&
     ((SystemState & CLEAR_LASER_FIRING_ARMED_CALIB_MASK)!=ARMED_MANUAL) &&
       !(SystemState & CALIB_ON))
    {
      printf("received Commands outside of Armed Manual State %x \n",cmd);
      return;
    } 

      switch(cmd)
        {
         case DEC_X:
              xCorrect++;
              break;
         case INC_X:
              xCorrect--;
              break;
         case DEC_Y:
              yCorrect--;
              break;
         case INC_Y:
              yCorrect++;
              break;

         default:
              printf("invalid command %x\n",cmd);
              break;
      }

}
//------------------------------------------------------------------------------------------------
// END static void ProcessCalibCommands
//------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------
// static void *NetworkInputThread
//------------------------------------------------------------------------------------------------
static void *NetworkInputThread(void *data)
{
 unsigned char Buffer[512];
 TMesssageHeader *MsgHdr;
 int fd=TcpConnectedPort->ConnectedFd,retval;
 
 SendSystemState(SystemState);

 while (1)
 {
   if ((retval=recv(fd, &Buffer, sizeof(TMesssageHeader),0)) != sizeof(TMesssageHeader)) 
     {
      if (retval==0) printf("Client Disconnnected\n");
      else printf("Connecton Lost %s\n", strerror(errno));
      break;
     }
   MsgHdr=(TMesssageHeader *)Buffer;
   MsgHdr->Len = ntohl(MsgHdr->Len);
   MsgHdr->Type = ntohl(MsgHdr->Type);

   if (MsgHdr->Len+sizeof(TMesssageHeader)>sizeof(Buffer))
     {
      printf("oversized message error %d\n",MsgHdr->Len);
      break;
     }
   if ((retval=recv(fd, &Buffer[sizeof(TMesssageHeader)],  MsgHdr->Len,0)) !=  MsgHdr->Len) 
     {
      if (retval==0) printf("Client Disconnnected\n");
      else printf("Connecton Lost %s\n", strerror(errno));
      break;
     }

   switch(MsgHdr->Type)
    {
      case MT_COMMANDS: 
      {
       TMesssageCommands *msgCmds=(TMesssageCommands *)Buffer;
       ProcessCommands(msgCmds->Commands);
      }
      break;
      case MT_CALIB_COMMANDS: 
      {
       TMesssageCalibCommands *msgCmds=(TMesssageCalibCommands *)Buffer;
       ProcessCalibCommands(msgCmds->Commands);
      }
      break;

      case MT_TARGET_SEQUENCE: 
      {
       TMesssageTargetOrder *msgTargetOrder=(TMesssageTargetOrder *)Buffer;
       ProcessFiringOrder(msgTargetOrder->FiringOrder);
      }
      break;
      case MT_PREARM: 
      {
       TMesssagePreArm *msgPreArm=(TMesssagePreArm *)Buffer;
       ProcessPreArm(msgPreArm->Code);
      }
      break;
      case MT_STATE_CHANGE_REQ: 
      {
       TMesssageChangeStateRequest *msgChangeStateRequest=(TMesssageChangeStateRequest *)Buffer;
       msgChangeStateRequest->State=(SystemState_t)ntohl(msgChangeStateRequest->State);

       ProcessStateChangeRequest(msgChangeStateRequest->State);
      }
      break;

      default:
       printf("Invalid Message Type\n");
      break; 
    }
  }
   isConnected=false;
   NetworkThreadID=-1; // Temp Fix OS probem determining if thread id are valid
   printf("Network Thread Exit\n");
   return NULL;
 }
//------------------------------------------------------------------------------------------------
// END static void *NetworkInputThread
//------------------------------------------------------------------------------------------------
//----------------------------------------------------------------
// Setup_Control_C_Signal_Handler_And_Keyboard_No_Enter - This 
// sets uo the Control-c Handler and put the keyboard in a mode
// where it will not
// 1. echo input
// 2. need enter hit to get a character 
// 3. block waiting for input
//-----------------------------------------------------------------
static void Setup_Control_C_Signal_Handler_And_Keyboard_No_Enter(void)
{
 struct sigaction sigIntHandler;
 sigIntHandler.sa_handler = Control_C_Handler; // Setup control-c callback 
 sigemptyset(&sigIntHandler.sa_mask);
 sigIntHandler.sa_flags = 0;
 sigaction(SIGINT, &sigIntHandler, NULL);
 ConfigKeyboardNoEnterBlockEcho();             // set keyboard configuration
}
//-----------------------------------------------------------------
// END Setup_Control_C_Signal_Handler_And_Keyboard_No_Enter
//-----------------------------------------------------------------
//----------------------------------------------------------------
// CleanUp - Performs cleanup processing before exiting the
// the program
//-----------------------------------------------------------------
static void CleanUp(void)
{
 void *res;
 int s;
 
RestoreKeyboard();                // restore Keyboard
 if (NetworkThreadID!=-1)
  {
   //printf("Cancel Network Thread\n");
   s = pthread_cancel(NetworkThreadID);
   if (s!=0)  printf("Network Thread Cancel Failure\n");
 
   //printf("Network Thread Join\n"); 
   s = pthread_join(NetworkThreadID, &res);
   if (s != 0)   printf("Network Thread Join Failure\n"); 

   if (res == PTHREAD_CANCELED)
       printf("Network Thread canceled\n"); 
   else
       printf("Network Thread was not canceled\n"); 
 }
 if (EngagementThreadID!=-1)
  {
   //printf("Cancel Engagement Thread\n");
   s = pthread_cancel(EngagementThreadID);
   if (s!=0)  printf("Engagement Thread Cancel Failure\n");
 
   //printf("Engagement Thread Join\n"); 
   s = pthread_join(EngagementThreadID, &res);
   if (s != 0)   printf("Engagement  Thread Join Failure\n"); 

   if (res == PTHREAD_CANCELED)
       printf("Engagement Thread canceled\n"); 
   else
       printf("Engagement Thread was not canceled\n"); 
 }

//  CloseCamera();
 CloseServos();
 
 laser(false);
 fire(false);
 calibrate(false);
 CloseGPIO();

 CloseTcpConnectedPort(&TcpConnectedPort); // Close network port;
 
 if (HaveOLED) ssd1306_end();
 printf("CleanUp Complete\n");
}
//-----------------------------------------------------------------
// END CleanUp
//-----------------------------------------------------------------
//----------------------------------------------------------------
// Control_C_Handler - called when control-c pressed
//-----------------------------------------------------------------
static void Control_C_Handler(int s)
{
 printf("Caught signal %d\n",s);
 CleanUp();
 printf("Exiting\n");
 exit(1);
}
//-----------------------------------------------------------------
// END Control_C_Handler
//-----------------------------------------------------------------
//----------------------------------------------------------------
// HandleInputChar - check if keys are press and proccess keys of
// interest.
//-----------------------------------------------------------------
static void HandleInputChar()
{
    int ch;
    if ((ch = getchar()) != EOF)
    {
        unsigned int cmd = 0;

        // 1) Map the pressed key to a command
        switch(ch)
        {
            case 'w':  
                cmd = PAN_UP_START;   
                break;
            case 's':  
                cmd = PAN_DOWN_START; 
                break;
            case 'a':  
                cmd = PAN_LEFT_START; 
                break;
            case 'd':  
                cmd = PAN_RIGHT_START;
                break;
            case 'f':  
                cmd = FIRE_START;     
                break;

            // Optionally, uppercase can be mapped to the STOP commands
            case 'W':  
                cmd = PAN_UP_STOP;    
                break;
            case 'S':  
                cmd = PAN_DOWN_STOP;  
                break;
            case 'A':  
                cmd = PAN_LEFT_STOP;  
                break;
            case 'D':  
                cmd = PAN_RIGHT_STOP; 
                break;
            case 'q':  
                cmd = FIRE_STOP;      
                break;

            default:
                cmd = 0; // no valid command
                break;
        }

        // 2) Execute the command using your existing switch-case structure
        switch(cmd)
        {
            case PAN_LEFT_START:
                RunCmds |= PAN_LEFT_START;
                RunCmds &= PAN_RIGHT_STOP; // stop opposite direction
                Pan += INC;
                ServoAngle(PAN_SERVO, Pan);
                break;

            case PAN_RIGHT_START:
                RunCmds |= PAN_RIGHT_START;
                RunCmds &= PAN_LEFT_STOP; // stop opposite direction
                Pan -= INC;
                ServoAngle(PAN_SERVO, Pan);
                break;

            case PAN_UP_START:
                RunCmds |= PAN_UP_START;
                RunCmds &= PAN_DOWN_STOP; // stop opposite direction
                Tilt += INC;
                ServoAngle(TILT_SERVO, Tilt);
                break;

            case PAN_DOWN_START:
                RunCmds |= PAN_DOWN_START;
                RunCmds &= PAN_UP_STOP; // stop opposite direction
                Tilt -= INC;
                ServoAngle(TILT_SERVO, Tilt);
                break;

            case FIRE_START:
                RunCmds |= FIRE_START;
                gpioWrite(29,true);
                break;

            case PAN_LEFT_STOP:
                RunCmds &= PAN_LEFT_STOP;
                break;

            case PAN_RIGHT_STOP:
                RunCmds &= PAN_RIGHT_STOP;
                break;

            case PAN_UP_STOP:
                RunCmds &= PAN_UP_STOP;
                break;

            case PAN_DOWN_STOP:
                RunCmds &= PAN_DOWN_STOP;
                break;

            case FIRE_STOP:
                RunCmds &= FIRE_STOP;
                gpioWrite(29,false);
                break;

            default:
                // You can decide how to handle an invalid command
                printf("invalid command %x\n", cmd);
                break;
        }
    }
}
//-----------------------------------------------------------------
// END HandleInputChar
//-----------------------------------------------------------------
//-----------------------------------------------------------------
// END of File
//-----------------------------------------------------------------

static void * EngagementCustom() 
{
  int ret;

  printf("Engagment in Progress\n");
  //laser(true);
  // usleep(100*10);
  // fire(true);
  gpioWrite(29,true);
  usleep(1000*200);
  // fire(false);
  gpioWrite(29,false);
  //laser(false);
  // armed(false);

  return NULL;
}

