/*
// The example of interoperability between OpenCL and OpenCV.
// This will loop through frames of video either from input media file
// or camera device and do processing of these data in OpenCL and then
// in OpenCV. In OpenCL it does inversion of pixels in left half of frame and
// in OpenCV it does blurring in the right half of frame.
*/
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS // eliminate build warning
#define CL_TARGET_OPENCL_VERSION 200  // 2.0

#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include "mediapipe/framework/formats/helpers.hpp"

#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace std;
using namespace cv;

OpenClWrapper::OpenClWrapper()
{
    cout << "\nPress ESC to exit\n" << endl;
    cout << "\n      'p' to toggle ON/OFF processing\n" << endl;
    cout << "\n       SPACE to switch between OpenCL buffer/image\n" << endl;

    m_camera_id  = 0; //cmd.get<int>("camera");
    m_file_name  = "0"; //cmd.get<string>("video");

    m_running    = false;
    m_process    = false;
    m_use_buffer = false;

    m_t0         = 0;
    m_t1         = 0;
    m_time       = 0.0;
    m_frequency  = (float)cv::getTickFrequency();

    m_context    = 0;
    m_device_id  = 0;
    m_queue      = 0;
    m_program    = 0;
    m_kernelBuf  = 0;
    m_kernelImg  = 0;
    m_img_src    = 0;
    m_mem_obj    = 0;
} // ctor


OpenClWrapper::~OpenClWrapper()
{
    if (m_queue)
    {
        clFinish(m_queue);
        clReleaseCommandQueue(m_queue);
        m_queue = 0;
    }

    if (m_program)
    {
        clReleaseProgram(m_program);
        m_program = 0;
    }

    if (m_img_src)
    {
        clReleaseMemObject(m_img_src);
        m_img_src = 0;
    }

    if (m_mem_obj)
    {
        clReleaseMemObject(m_mem_obj);
        m_mem_obj = 0;
    }

    if (m_kernelBuf)
    {
        clReleaseKernel(m_kernelBuf);
        m_kernelBuf = 0;
    }

    if (m_kernelImg)
    {
        clReleaseKernel(m_kernelImg);
        m_kernelImg = 0;
    }

    if (m_device_id)
    {
        clReleaseDevice(m_device_id);
        m_device_id = 0;
    }

    if (m_context)
    {
        clReleaseContext(m_context);
        m_context = 0;
    }
} // dtor


int OpenClWrapper::initOpenCL()
{
    cl_int res = CL_SUCCESS;
    cl_uint num_entries = 0;

    res = clGetPlatformIDs(0, 0, &num_entries);
    if (CL_SUCCESS != res)
        return -1;

    m_platform_ids.resize(num_entries);

    res = clGetPlatformIDs(num_entries, &m_platform_ids[0], 0);
    if (CL_SUCCESS != res)
        return -1;

    unsigned int i;

    // create context from first platform with GPU device
    for (i = 0; i < m_platform_ids.size(); i++)
    {
        cl_context_properties props[] =
        {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(m_platform_ids[i]),
            0
        };

        m_context = clCreateContextFromType(props, CL_DEVICE_TYPE_GPU, 0, 0, &res);
        if (0 == m_context || CL_SUCCESS != res)
            continue;

        res = clGetContextInfo(m_context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &m_device_id, 0);
        if (CL_SUCCESS != res)
            return -1;

        m_queue = clCreateCommandQueue(m_context, m_device_id, 0, &res);
        if (0 == m_queue || CL_SUCCESS != res)
            return -1;

        const char* kernelSrc =
            "__kernel "
            "void bitwise_inv_buf_8uC1("
            "    __global unsigned char* pSrcDst,"
            "             int            srcDstStep,"
            "             int            rows,"
            "             int            cols)"
            "{"
            "    int x = get_global_id(0);"
            "    int y = get_global_id(1);"
            "    int idx = mad24(y, srcDstStep, x);"
            "    pSrcDst[idx] = ~pSrcDst[idx];"
            "}"
            "__kernel "
            "void bitwise_inv_img_8uC1("
            "    read_only  image2d_t srcImg,"
            "    write_only image2d_t dstImg)"
            "{"
            "    int x = get_global_id(0);"
            "    int y = get_global_id(1);"
            "    int2 coord = (int2)(x, y);"
            "    uint4 val = read_imageui(srcImg, coord);"
            "    val.x = (~val.x) & 0x000000FF;"
            "    write_imageui(dstImg, coord, val);"
            "}";
        size_t len = strlen(kernelSrc);
        m_program = clCreateProgramWithSource(m_context, 1, &kernelSrc, &len, &res);
        if (0 == m_program || CL_SUCCESS != res)
            return -1;

        res = clBuildProgram(m_program, 1, &m_device_id, 0, 0, 0);
        if (CL_SUCCESS != res)
            return -1;

        m_kernelBuf = clCreateKernel(m_program, "bitwise_inv_buf_8uC1", &res);
        if (0 == m_kernelBuf || CL_SUCCESS != res)
            return -1;

        m_kernelImg = clCreateKernel(m_program, "bitwise_inv_img_8uC1", &res);
        if (0 == m_kernelImg || CL_SUCCESS != res)
            return -1;

        m_platformInfo.QueryInfo(m_platform_ids[i]);
        m_deviceInfo.QueryInfo(m_device_id);

        // attach OpenCL context to OpenCV
        cv::ocl::attachContext(m_platformInfo.Name(), m_platform_ids[i], m_context, m_device_id);

        break;
    }

    return m_context != 0 ? CL_SUCCESS : -1;
} // initOpenCL()


// this function is an example of "typical" OpenCL processing pipeline
// It creates OpenCL buffer or image, depending on use_buffer flag,
// from input media frame and process these data
// (inverts each pixel value in half of frame) with OpenCL kernel
int OpenClWrapper::process_frame_with_open_cl(cv::Mat& frame, bool use_buffer, cl_mem* mem_obj)
{
    cl_int res = CL_SUCCESS;

    CV_Assert(mem_obj);

    cl_kernel kernel = 0;
    cl_mem mem = mem_obj[0];

    if (0 == mem || 0 == m_img_src)
    {
        // allocate/delete cl memory objects every frame for the simplicity.
        // in real application more efficient pipeline can be built.

        if (use_buffer)
        {
            cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;

            mem = clCreateBuffer(m_context, flags, frame.total(), frame.ptr(), &res);
            if (0 == mem || CL_SUCCESS != res)
                return -1;

            res = clSetKernelArg(m_kernelBuf, 0, sizeof(cl_mem), &mem);
            if (CL_SUCCESS != res)
                return -1;

            res = clSetKernelArg(m_kernelBuf, 1, sizeof(int), &frame.step[0]);
            if (CL_SUCCESS != res)
                return -1;

            res = clSetKernelArg(m_kernelBuf, 2, sizeof(int), &frame.rows);
            if (CL_SUCCESS != res)
                return -1;

            int cols2 = frame.cols / 2;
            res = clSetKernelArg(m_kernelBuf, 3, sizeof(int), &cols2);
            if (CL_SUCCESS != res)
                return -1;

            kernel = m_kernelBuf;
        }
        else
        {
            cl_mem_flags flags_src = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;

            cl_image_format fmt;
            fmt.image_channel_order     = CL_R;
            fmt.image_channel_data_type = CL_UNSIGNED_INT8;

            cl_image_desc desc_src;
            desc_src.image_type        = CL_MEM_OBJECT_IMAGE2D;
            desc_src.image_width       = frame.cols;
            desc_src.image_height      = frame.rows;
            desc_src.image_depth       = 0;
            desc_src.image_array_size  = 0;
            desc_src.image_row_pitch   = frame.step[0];
            desc_src.image_slice_pitch = 0;
            desc_src.num_mip_levels    = 0;
            desc_src.num_samples       = 0;
            desc_src.buffer            = 0;
            m_img_src = clCreateImage(m_context, flags_src, &fmt, &desc_src, frame.ptr(), &res);
            if (0 == m_img_src || CL_SUCCESS != res)
                return -1;

            cl_mem_flags flags_dst = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;

            cl_image_desc desc_dst;
            desc_dst.image_type        = CL_MEM_OBJECT_IMAGE2D;
            desc_dst.image_width       = frame.cols;
            desc_dst.image_height      = frame.rows;
            desc_dst.image_depth       = 0;
            desc_dst.image_array_size  = 0;
            desc_dst.image_row_pitch   = 0;
            desc_dst.image_slice_pitch = 0;
            desc_dst.num_mip_levels    = 0;
            desc_dst.num_samples       = 0;
            desc_dst.buffer            = 0;
            mem = clCreateImage(m_context, flags_dst, &fmt, &desc_dst, 0, &res);
            if (0 == mem || CL_SUCCESS != res)
                return -1;

            size_t origin[] = { 0, 0, 0 };
            size_t region[] = { (size_t)frame.cols, (size_t)frame.rows, 1 };
            cl_event asyncEvent = 0;
            res = clEnqueueCopyImage(m_queue, m_img_src, mem, origin, origin, region, 0, 0, &asyncEvent);
            if (CL_SUCCESS != res)
                return -1;

            res = clWaitForEvents(1, &asyncEvent);
            clReleaseEvent(asyncEvent);
            if (CL_SUCCESS != res)
                return -1;

            res = clSetKernelArg(m_kernelImg, 0, sizeof(cl_mem), &m_img_src);
            if (CL_SUCCESS != res)
                return -1;

            res = clSetKernelArg(m_kernelImg, 1, sizeof(cl_mem), &mem);
            if (CL_SUCCESS != res)
                return -1;

            kernel = m_kernelImg;
        }
    }

    // process left half of frame in OpenCL
    size_t size[] = { (size_t)frame.cols / 2, (size_t)frame.rows };
    cl_event asyncEvent = 0;
    res = clEnqueueNDRangeKernel(m_queue, kernel, 2, 0, size, 0, 0, 0, &asyncEvent);
    if (CL_SUCCESS != res)
        return -1;

    res = clWaitForEvents(1, &asyncEvent);
    clReleaseEvent(asyncEvent);
    if (CL_SUCCESS != res)
        return -1;

    mem_obj[0] = mem;

    return  0;
}


// this function is an example of interoperability between OpenCL buffer
// and OpenCV UMat objects. It converts (without copying data) OpenCL buffer
// to OpenCV UMat and then do blur on these data
int OpenClWrapper::process_cl_buffer_with_opencv(cl_mem buffer, size_t step, int rows, int cols, int type, cv::UMat& u)
{
    cv::ocl::convertFromBuffer(buffer, step, rows, cols, type, u);

    // process right half of frame in OpenCV
    cv::Point pt(u.cols / 2, 0);
    cv::Size  sz(u.cols / 2, u.rows);
    cv::Rect roi(pt, sz);
    cv::UMat uroi(u, roi);
    cv::blur(uroi, uroi, cv::Size(7, 7), cv::Point(-3, -3));

    if (buffer)
        clReleaseMemObject(buffer);
    m_mem_obj = 0;

    return 0;
}


// this function is an example of interoperability between OpenCL image
// and OpenCV UMat objects. It converts OpenCL image
// to OpenCV UMat and then do blur on these data
int OpenClWrapper::process_cl_image_with_opencv(cl_mem image, cv::UMat& u)
{
    cv::ocl::convertFromImage(image, u);

    // process right half of frame in OpenCV
    cv::Point pt(u.cols / 2, 0);
    cv::Size  sz(u.cols / 2, u.rows);
    cv::Rect roi(pt, sz);
    cv::UMat uroi(u, roi);
    cv::blur(uroi, uroi, cv::Size(7, 7), cv::Point(-3, -3));

    if (image)
        clReleaseMemObject(image);
    m_mem_obj = 0;

    if (m_img_src)
        clReleaseMemObject(m_img_src);
    m_img_src = 0;

    return 0;
}


int OpenClWrapper::run()
{
    if (0 != initOpenCL())
        return -1;

    if (0 != initVideoSource())
        return -1;

    Mat img_to_show;

    // set running state until ESC pressed
    setRunning(true);
    // set process flag to show some data processing
    // can be toggled on/off by 'p' button
    setDoProcess(true);
    // set use buffer flag,
    // when it is set to true, will demo interop opencl buffer and cv::Umat,
    // otherwise demo interop opencl image and cv::UMat
    // can be switched on/of by SPACE button
    setUseBuffer(true);

    // Iterate over all frames
    while (isRunning() && nextFrame(m_frame))
    {
        cv::cvtColor(m_frame, m_frameGray, COLOR_BGR2GRAY);

        UMat uframe;

        // work
        timerStart();

        if (doProcess())
        {
            process_frame_with_open_cl(m_frameGray, useBuffer(), &m_mem_obj);

            if (useBuffer())
                process_cl_buffer_with_opencv(
                    m_mem_obj, m_frameGray.step[0], m_frameGray.rows, m_frameGray.cols, m_frameGray.type(), uframe);
            else
                process_cl_image_with_opencv(m_mem_obj, uframe);
        }
        else
        {
            m_frameGray.copyTo(uframe);
        }

        timerEnd();

        uframe.copyTo(img_to_show);

        putText(img_to_show, "Version : " + m_platformInfo.Version(), Point(5, 30), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
        putText(img_to_show, "Name : " + m_platformInfo.Name(), Point(5, 60), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
        putText(img_to_show, "Device : " + m_deviceInfo.Name(), Point(5, 90), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
        cv::String memtype = useBuffer() ? "buffer" : "image";
        putText(img_to_show, "interop with OpenCL " + memtype, Point(5, 120), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
        putText(img_to_show, "Time : " + timeStr() + " msec", Point(5, 150), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);

        imshow("opencl_interop", img_to_show);

        handleKey((char)waitKey(3));
    }

    return 0;
}

