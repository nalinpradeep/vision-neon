
/**
 Copyright (c) 2011 Nalin Senthamil
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 **/

#ifndef _arm_neon_operations_h_
#define _arm_neon_operations_h_

#include <opencv/cv.h>

namespace vision{
    void ad_copyImage_neon32(const cv::Mat& src,cv::Mat& dst);
    void ad_copyImage_neon32(unsigned char* src,unsigned int step,unsigned int height,unsigned char*dst);
	
    void ad_convertToGray_neon32(const cv::Mat& src,cv::Mat& dst);
    
    void ad_downsampleBy2_BGRA_neon64(const cv::Mat& src,cv::Mat& dst);
    void ad_downsampleBy2_Gray_neon32(const cv::Mat& src,cv::Mat& dst);
    void ad_downsampleBy4_Gray_neon64(const cv::Mat& src,cv::Mat& dst);
}

#endif