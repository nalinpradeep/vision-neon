
#include "arm_neon_operations.h"
#ifdef __ARM_NEON__
#include "arm_neon.h"
#endif

#define ASSERT_ARM_NEON_EQUAL_SIZE(i,j) (assert(i==j))
#define ASSERT_ARM_NEON_MOD_SIZE(i,j) (assert(i%j==0))

using namespace cv;

namespace vision {

#ifdef __ARM_NEON__
void copyImage_neon32(unsigned char* src,unsigned int step,unsigned int height,unsigned char*dst){
	unsigned char* data = src; 
    unsigned char* out = dst; 
    unsigned int _isize = step*height,i;
    ASSERT_ARM_NEON_MOD_SIZE(step,32);
    for(i=0;i<_isize;i+=32,data+=32,out+=32){
        asm volatile (
                      "vld1.8 {q0,q1},[%0]	\n\t"
                      "vst1.8 {q0,q1},[%1]	\n\t"
                      :: "r"(data),"r"(out)
                      :"r0","r14", "cc", "memory", "q0", "q1"
                      );	
    }
}

void copyImage_neon32(const cv::Mat& src,cv::Mat& dst){
	if(dst.empty())
		dst.create(src.rows,src.cols,src.type());
	copyImage_neon32(src.data,src.step,src.rows,dst.data);
}

    
void convertToGray_neon32(const cv::Mat& src,cv::Mat& dst){
	if(dst.empty()) 
		dst.create(src.rows,src.cols,CV_8UC1);
    unsigned int _isize = src.step*src.rows,i;
	unsigned char* data = src.data;
	unsigned char* out = dst.data;
	for(i=0;i<_isize;i+=32,data+=32,out+=8){
		asm volatile (
                      "mov         r4, #28        \n"
                      "mov         r5, #151       \n"
                      "mov         r6, #77        \n"
                      "vdup.8      d4, r4         \n"
                      "vdup.8      d5, r5         \n"
                      "vdup.8      d6, r6         \n"
                      
                      "vld4.8 {d0,d1,d2,d3}, [%0]    \n"
                      "vmull.u8    q4,d0,d4       \n"
                      "vmlal.u8    q4,d1,d5       \n"
                      "vmlal.u8    q4,d2,d6       \n"
                      
                      "vshrn.u16   d7, q4,#8     \n"
                      "vst1.8      {d7},[%1]    \n"
                      :: "r"(data),"r"(out)
                      :"r0","r4","r5","r6","cc", "memory", "q0", "q1", "q2", "q3", "q4"
                      );   
	}
}
	
    void downsampleBy2_BGRA_neon64(const cv::Mat& src,cv::Mat& dst){
        unsigned int x,x32,y;
        unsigned int _h=src.rows,_s=src.step,_w = src.cols<<2;
        unsigned int _2s = _s<<1;
        if(dst.empty())
            dst.create(src.rows>>1,src.cols>>1,CV_8UC4);
        
        unsigned char* data0 = src.data;
        unsigned char* data1;
        unsigned char* out = dst.data;
        
        ASSERT_ARM_NEON_MOD_SIZE(src.step,64);
        ASSERT_ARM_NEON_MOD_SIZE(dst.step,32);
        
        for(y=0;y<_h;y+=2,data0+=_2s){
            data1 = data0 + _s;
            for(x=0,x32=32;x<_w;x+=64,x32+=64,out+=32){
                asm volatile (
                              "vld4.8 {d0,d2,d4,d6},[%0]	\n\t"
                              "vld4.8 {d1,d3,d5,d7},[%1]	\n\t"
                              "vld4.8 {d8,d10,d12,d14},[%2]	\n\t"
                              "vld4.8 {d9,d11,d13,d15},[%3]	\n\t"
                              
                              "vpaddl.u8 q0,q0 \n\t"
                              "vpaddl.u8 q1,q1 \n\t"
                              "vpaddl.u8 q2,q2 \n\t"
                              
                              "vpaddl.u8 q4,q4 \n\t"
                              "vpaddl.u8 q5,q5 \n\t"
                              "vpaddl.u8 q6,q6 \n\t"
                              
                              "vadd.u16 q0,q0,q4 \n\t"
                              "vadd.u16 q1,q1,q5 \n\t"
                              "vadd.u16 q2,q2,q6 \n\t"
                              
                              "vshrn.u16 d0,q0,#2 \n\t"
                              "vshrn.u16 d1,q1,#2 \n\t"
                              "vshrn.u16 d2,q2,#2 \n\t"
                              
                              //"mov r3,#0 \n\t"
                              //"vdup.8 d3,r3 \n\t"
                              "vmov d3,d6 \n\t"
                              
                              "vst4.8 {d0,d1,d2,d3},[%4] \n\t"
                              :: "r"(data0+x),"r"(data0+x32),"r"(data1+x),"r"(data1+x32),"r"(out)
                              :"r0","r14", "cc", "memory", "q0", "q1", "q2", "q3","q4","q5","q6","q7","q8"
                              );   
            }
        }
    }
    
	
    void downsampleBy2_Gray_neon32(const cv::Mat& src,cv::Mat& dst){
        size_t x,y;
        size_t _w = src.cols,_h=src.rows,_s=src.step;
        size_t _2s = _s<<1;
        if(dst.empty())
            dst.create(src.rows>>1,src.cols>>1,CV_8UC1);
        unsigned char* data0 = src.data;
        unsigned char* data1;
        unsigned char* out = dst.data;
        
        ASSERT_ARM_NEON_MOD_SIZE(src.step,32);
        ASSERT_ARM_NEON_MOD_SIZE(dst.step,16);
        for(y=0;y<_h;y+=2,data0+=_2s){
            data1 = data0 + _s;
            for(x=0;x<_w;x+=32,out+=16){
                asm volatile (
                              "vld1.8 {q0,q1},[%0] \n\t"
                              "vld1.8 {q2,q3},[%1] \n\t"
                              
                              "vpaddl.u8 q0,q0 \n\t"
                              "vpaddl.u8 q1,q1 \n\t"
                              "vpaddl.u8 q2,q2 \n\t"
                              "vpaddl.u8 q3,q3 \n\t"
                              
                              "vadd.u16 q0,q0,q2 \n\t"
                              "vadd.u16  q1,q1,q3 \n\t"
                              
                              "vshrn.u16 d0,q0,#2 \n\t"
                              "vshrn.u16 d1,q1,#2 \n\t"
                              
                              "vst1.8 {q0},[%2] \n\t"
                              
                              :: "r"(data0+x),"r"(data1+x),"r"(out)
                              :"r0","r14", "cc", "memory", "q0", "q1", "q2", "q3"
                              );	
            }
        }
    }

    void downsampleBy4_Gray_neon64(const cv::Mat& src,cv::Mat& dst){
        unsigned int x,y,x32;
        unsigned int _w = src.cols,_h=src.rows,_s=src.step;
        unsigned int _4s = _s<<2;
        if(dst.empty()){
            unsigned int _w32 = src.cols>>2;
            unsigned int mod = (_w32%32);
            if(mod==0)
                dst.create(src.rows>>2,_w32,CV_8UC1);
            else
                dst = cv::Mat_<uchar>::zeros(src.rows>>2,_w32+(32-mod));
        }
        size_t extra_offset= (dst.cols-(src.cols>>2));   
        //std::cerr<<"dst-"<<dst.cols<<",offset-"<<extra_offset<<std::endl;
        unsigned char* data0 = src.data;
        unsigned char* data1;
        unsigned char* data2;
        unsigned char* data3;
        unsigned char* out = dst.data;
        unsigned int _w64 = (_w/64)*64;
    
        ASSERT_ARM_NEON_MOD_SIZE(_w64,64);
        ASSERT_ARM_NEON_MOD_SIZE(dst.cols,32);
    
        for(y=0;y<_h;y+=4,data0+=_4s,out+=extra_offset){
            data1 = data0 + _s;
            data2 = data1 + _s;
            data3 = data2 + _s;
            for(x=0,x32=32;x<_w64;x+=64,x32+=64,out+=16){
                asm volatile (
                          "vld1.8 {q0,q1},[%0] \n\t"
                          "vld1.8 {q4,q5},[%1] \n\t"
                          "vld1.8 {q8,q9},[%2] \n\t"
                          "vld1.8 {q12,q13},[%3] \n\t"
                          
                          "vld1.8 {q2,q3},[%4] \n\t"
                          "vld1.8 {q6,q7},[%5] \n\t"
                          "vld1.8 {q10,q11},[%6] \n\t"
                          "vld1.8 {q14,q15},[%7] \n\t"
                          
                          "vpaddl.u8 q0,q0 \n\t"
                          "vpaddl.u8 q1,q1 \n\t"
                          "vpaddl.u8 q2,q2 \n\t"
                          "vpaddl.u8 q3,q3 \n\t"
                          
                          "vpadal.u8 q0,q4 \n\t"
                          "vpadal.u8 q1,q5 \n\t"
                          "vpadal.u8 q2,q6 \n\t"
                          "vpadal.u8 q3,q7 \n\t"
                          
                          "vpadal.u8 q0,q8 \n\t"
                          "vpadal.u8 q1,q9 \n\t"
                          "vpadal.u8 q2,q10 \n\t"
                          "vpadal.u8 q3,q11 \n\t"
                          
                          "vpadal.u8 q0,q12 \n\t"
                          "vpadal.u8 q1,q13 \n\t"
                          "vpadal.u8 q2,q14 \n\t"
                          "vpadal.u8 q3,q15 \n\t"
                          
                          "vpadd.u16 d0,d0,d1 \n\t"
                          "vpadd.u16 d1,d2,d3 \n\t"
                          "vpadd.u16 d2,d4,d5 \n\t"
                          "vpadd.u16 d3,d6,d7 \n\t"
                          
                          "vshrn.u16 d0,q0,#4 \n\t"
                          "vshrn.u16 d1,q1,#4 \n\t"
                          
                          "vst1.8 {q0},[%8] \n\t"
                          
                          :: "r"(data0+x),"r"(data1+x),"r"(data2+x),"r"(data3+x),
                          "r"(data0+x32),"r"(data1+x32),"r"(data2+x32),"r"(data3+x32),
                          "r"(out)
                          :"r0","r14","cc", "memory", 
                          "q0", "q1", "q2", "q3","q4", "q5", "q6", "q7",
                          "q8", "q9", "q10", "q11",
                          "q12", "q13", "q14", "q15"
                          );	
            }
        
            for(;x<_w;x+=32,out+=8){
                asm volatile (
                            "vld1.8 {q0,q1},[%0] \n\t"
                            "vld1.8 {q4,q5},[%1] \n\t"
                            "vld1.8 {q8,q9},[%2] \n\t"
                            "vld1.8 {q12,q13},[%3] \n\t"
                            
                            "vpaddl.u8 q0,q0 \n\t"
                            "vpaddl.u8 q1,q1 \n\t"
                            
                            "vpadal.u8 q0,q4 \n\t"
                            "vpadal.u8 q1,q5 \n\t"
                            
                            "vpadal.u8 q0,q8 \n\t"
                            "vpadal.u8 q1,q9 \n\t"
                              
                            "vpadal.u8 q0,q12 \n\t"
                            "vpadal.u8 q1,q13 \n\t"
                            
                            "vpadd.u16 d0,d0,d1 \n\t"
                            "vpadd.u16 d1,d2,d3 \n\t"
                            
                            "vshrn.u16 d0,q0,#4 \n\t"
                            
                            "vst1.8 {d0},[%4] \n\t"
                            
                            :: "r"(data0+x),"r"(data1+x),"r"(data2+x),"r"(data3+x),
                            "r"(out)
                            :"r0","r14","cc", "memory", 
                            "q0", "q1","q4", "q5", "q8", "q9", "q12", "q13"
                            );
            }
        }
    }
#endif
}
