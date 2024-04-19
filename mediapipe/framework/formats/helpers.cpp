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
#include <mutex>
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
static std::mutex g_dump_mutex;
static std::mutex g_dump_mutex2;

void dumpMatToFile(std::string& fileName, cv::UMat& umat){
    std::lock_guard<std::mutex> guard(g_dump_mutex);
    cv::FileStorage file(fileName, cv::FileStorage::WRITE);
    cv::Mat input;
    //file << "camera_frame_raw" << camera_frame_raw;
    umat.copyTo(input);
    file << "input_frame_mat" << input;

    file.release();
}

void dumpMatToFile(std::string& fileName, cv::Mat& umat){
    std::lock_guard<std::mutex> guard(g_dump_mutex2);
    cv::FileStorage file(fileName, cv::FileStorage::WRITE);
    file << "input_frame_mat" << umat;

    file.release();
}

cl_channel_order GetChannelOrderFromMatType(int cvMatType) {
    switch (cvMatType) {
        case CV_8UC1:
        case CV_8SC1:
            std::cout<< "Using CL_R " << endl;
            return CL_R;
        case CV_8UC2:
        case CV_8SC2:
            std::cout<< "Using CL_RG " << endl;
            return CL_RG;
        case CV_8UC3:
        case CV_8SC3:
            std::cout<< "Using CL_RGB " << endl;
            return CL_RGB;
        case CV_8UC4:
        case CV_8SC4:
            std::cout<< "Using CL_RGBA " << endl;
            return CL_RGBA;
        default:
        {
            std::cout<< "Using default CL_R for CV type: " << cvMatType << endl;
            return CL_R; // Default to single channel
        }
    }
}


cl_channel_type GetChannelDataTypeFromOrder(cl_channel_order cl_order) {
    switch (cl_order) {
    case CL_R:
    case CL_A:
        return CL_UNSIGNED_INT8; // Or your desired data type
        std::cout<< "Using CL_UNSIGNED_INT8 " << endl;
    case CL_RG:
        return CL_UNSIGNED_INT16; // Or your desired data type
    case CL_RGB:
        std::cout<< "Using CL_HALF_FLOAT " << endl;
        return CL_HALF_FLOAT; // Or your desired data type
    case CL_RGBA:
        std::cout<< "Using CL_FLOAT " << endl;
        return CL_FLOAT; // Or your desired data type
    // Add more cases for other channel orders as needed
    default:
        // Handle unsupported channel order
        std::cout<< "Using default CL_UNSIGNED_INT8 for channel order: " << cl_order << endl;
        return CL_UNSIGNED_INT8;
        break;
    }
}

const char* clGetErrorString(int errorCode) {
    switch (errorCode) {
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
        case -69: return "CL_INVALID_PIPE_SIZE";
        case -70: return "CL_INVALID_DEVICE_QUEUE";
        case -71: return "CL_INVALID_SPEC_ID";
        case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
        case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
        case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
        case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
        case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
        case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
        case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
        case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
        case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
        case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
        case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
        case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
        case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
        case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
        case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
        case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
        case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
        case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
        default: return "CL_UNKNOWN_ERROR";
    }
}

OpenClWrapper::OpenClWrapper()
{
    m_camera_id  = 0; //cmd.get<int>("camera");
    m_file_name  = "0"; //cmd.get<string>("video");

    m_is_initialized = false;
    m_running    = false;
    m_process    = false;
    m_use_buffer = false;

    m_t0         = 0;
    m_t1         = 0;
    m_time       = 0.0;
    m_frequency  = (float)cv::getTickFrequency();

    m_context    = 0;
    //m_device_id  = 0;
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

    /*if (m_device_id)
    {
        clReleaseDevice(m_device_id);
        m_device_id = 0;
    }*/

    if (m_context)
    {
        clReleaseContext(m_context);
        m_context = 0;
    }
} // dtor

bool OpenClWrapper::m_is_initialized = false;
cl_context OpenClWrapper::m_context = nullptr;
cl_command_queue OpenClWrapper::m_queue = nullptr;

int OpenClWrapper::initOpenCL()
{
    if (m_is_initialized){
        std::cout << "OpenCL already initialized." << std::endl;
        return CL_SUCCESS;
    }

    cl_int res = CL_SUCCESS;
    cl_uint num_entries = 0;

    res = clGetPlatformIDs(0, 0, &num_entries);
    if (CL_SUCCESS != res)
        return -1;

    opencl::PlatformInfo        m_platformInfo;
    opencl::DeviceInfo          m_deviceInfo;
    cl_device_id                m_device_id;

    std::vector<cl_platform_id> m_platform_ids;

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

        cl_device_id device;
        // Get the first available device on the platform
        res = clGetDeviceIDs(m_platform_ids[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (res != CL_SUCCESS) {
            printf("Error getting device IDs\n");
            return 1;
        }

        // Query the device for supported image formats
        cl_image_format supportedFormats[128]; // Assuming maximum 128 supported formats
        cl_uint numSupportedFormats;
        
        m_context = clCreateContextFromType(props, CL_DEVICE_TYPE_GPU, 0, 0, &res);
        if (0 == m_context || CL_SUCCESS != res)
            continue;

        res = clGetContextInfo(m_context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &m_device_id, 0);
        if (CL_SUCCESS != res)
            return -1;

        res = clGetSupportedImageFormats(m_context, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, 128, supportedFormats, &numSupportedFormats);
        if (res != CL_SUCCESS) {
            printf("Error getting supported image formats\n");
            return 1;
        }

        // Check if CL_HALF_FLOAT is supported
        cl_bool supportsHalfFloat = CL_FALSE;
        for (cl_uint i = 0; i < numSupportedFormats; ++i) {
            if (supportedFormats[i].image_channel_data_type == CL_HALF_FLOAT) {
                supportsHalfFloat = CL_TRUE;
                break;
            }
        }

        // Print the result
        if (supportsHalfFloat == CL_TRUE) {
            printf("Device supports CL_HALF_FLOAT format\n");
        } else {
            printf("Device does not support CL_HALF_FLOAT format\n");
        }

        m_queue = clCreateCommandQueue(m_context, m_device_id, 0, &res);
        if (0 == m_queue || CL_SUCCESS != res)
            return -1;

        m_platformInfo.QueryInfo(m_platform_ids[i]);
        m_deviceInfo.QueryInfo(m_device_id);

        // attach OpenCL context to OpenCV
        cv::ocl::attachContext(m_platformInfo.Name(), m_platform_ids[i], m_context, m_device_id);

        break;
    }

    m_is_initialized = true;
    std::cout << "OpenCL initialized for the first time." << std::endl;
    cout << "Version : " << m_platformInfo.Version() << std::endl;
    cout << "Name : " << m_platformInfo.Name()<< std::endl;
    cout <<  "Device : " << m_deviceInfo.Name()<< std::endl;

    if (m_device_id)
    {
        clReleaseDevice(m_device_id);
        m_device_id = 0;
    }

    //printInfo();
    return m_context != 0 ? CL_SUCCESS : -1;
} // initOpenCL()

int OpenClWrapper::createMemObject(cl_mem* mem_obj, cv::UMat& inputData){
    //cv::ocl::Image2D image = cv::ocl::Image2D(inputData);
    //mem_obj[0] = image.handle;

    return 0;
}
int OpenClWrapper::createMemObject(cl_mem* mem_obj, cv::Mat& inputData){
    // OLD IMPLEMENTATION
    cl_int res = CL_SUCCESS;
    cl_mem mem = mem_obj[0];

    if (inputData.ptr() == nullptr)
    {
        std::cout << "Error:createMemObject nupptr as input ptr" << std::endl;
        return -1;
    }

    cl_image_format fmt;
    cl_channel_order channelOrder = GetChannelOrderFromMatType(inputData.type());
    fmt.image_channel_order     = channelOrder;
    cl_channel_type channelType = GetChannelDataTypeFromOrder(channelOrder);
    fmt.image_channel_data_type = channelType;

    cl_mem_flags flags_dst = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
    // TODO: fix CL_INVALID_IMAGE_FORMAT_DESCRIPTOR now - Hardcoded values
    // fmt.image_channel_order     = CL_R;
    // fmt.image_channel_data_type = CL_UNSIGNED_INT8;

    cl_image_desc desc_dst;
    memset(&desc_dst, 0, sizeof(cl_image_desc));
    desc_dst.image_type        = CL_MEM_OBJECT_IMAGE2D;
    desc_dst.image_width       = inputData.cols;
    desc_dst.image_height      = inputData.rows;
    desc_dst.image_depth       = 0;
    desc_dst.image_array_size  = 0;
    // desc_dst.image_row_pitch   = inputData.step[0];
    desc_dst.image_row_pitch   = inputData.step[0];
    desc_dst.image_slice_pitch = 0;
    desc_dst.num_mip_levels    = 0;
    desc_dst.num_samples       = 0;
    desc_dst.buffer            = 0;
    mem = clCreateImage(m_context, flags_dst, &fmt, &desc_dst, inputData.ptr(), &res);
    if (0 == mem || CL_SUCCESS != res){
        std::cout <<"Error: " << clGetErrorString(res) << std::endl;
        return -1;
    }

    mem_obj[0] = mem;

    return 0;
}

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

    //if (0 != initVideoSource())
    //    return -1;

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
        //timerStart();

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

        //timerEnd();

        uframe.copyTo(img_to_show);

        imshow("opencl_interop", img_to_show);

        //handleKey((char)waitKey(3));
    }

    return 0;
}

void OpenClWrapper::printInfo()
{
    // TODO
}