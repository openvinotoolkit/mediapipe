#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";

std::mutex g_mutex;
std::condition_variable g_cv;
std::atomic<int> initializedThreads = 0;
bool g_start = false;

absl::Status RunGraph(const std::string &graphPath,
                      const std::string &input_video_path,
                      const std::string &output_video_path) {
  // PHASE 1: Setup - Load graph config ///////////////////////////////////////////////////////
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      graphPath, &calculator_graph_config_contents));
  ABSL_LOG(INFO) << "Get calculator graph config contents: "
                 << calculator_graph_config_contents;

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  ABSL_LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  initializedThreads++;
  cv::VideoCapture capture;
  capture.open(input_video_path);
  RET_CHECK(capture.isOpened()) << "Failed to open " << input_video_path;

  cv::VideoWriter writer;

  // Start graph
  ABSL_LOG(INFO) << "Start running the calculator graph.";
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                      graph.AddOutputStreamPoller(kOutputStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));
  // PHASE 3: Timed frame processing //////////////////////////////////////////////////////////
  ABSL_LOG(INFO) << "Start grabbing and processing frames.";

  int count_frames = 0;
  auto begin = std::chrono::high_resolution_clock::now();   // ← starts HERE
  // All threads block here until main() calls cv.notify_all(),
  // so frame processing starts at the same wall-clock instant.
  {
    std::unique_lock<std::mutex> lock(g_mutex);
    g_cv.wait(lock, [] { return g_start; });
  }
  // Open video
  ABSL_LOG(INFO) << "Initialize the video.";
  while (true) {
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;

    if (camera_frame_raw.empty()) {
      ABSL_LOG(INFO) << "Empty frame, end of video reached.";
      break;
    }

    count_frames++;

    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

    // Wrap Mat into ImageFrame
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);

    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send packet
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

    MP_RETURN_IF_ERROR(
        graph.AddPacketToInputStream(
            kInputStream,
            mediapipe::Adopt(input_frame.release())
                .At(mediapipe::Timestamp(frame_timestamp_us))));

    // Receive output
    mediapipe::Packet packet;
    if (!poller.Next(&packet))
      break;

    auto &output_frame = packet.Get<mediapipe::ImageFrame>();
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

    // Initialize writer lazily
    if (!writer.isOpened()) {
      ABSL_LOG(INFO) << "Prepare video writer.";
      writer.open(
          output_video_path,
          mediapipe::fourcc('a', 'v', 'c', '1'),
          capture.get(cv::CAP_PROP_FPS),
          output_frame_mat.size());
      RET_CHECK(writer.isOpened());
    }

    writer.write(output_frame_mat);
  }

  // PHASE 4: Report //////////////////////////////////////////////////////////////////// 
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - begin);

  auto totalTime = duration.count();
  float avgFps = (1000000.0f * count_frames) / totalTime;
  float avgLatencyms = 1000.0f / avgFps;

  LOG(INFO) << "[" << graphPath << "] "
            << "Frames:" << count_frames
            << ", Duration [ms]:" << totalTime / 1000
            << ", FPS:" << avgFps
            << ", Avg latency [ms]:" << avgLatencyms;

  LOG(INFO) << "Shutting down.";
  if (writer.isOpened())
    writer.release();

  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char **argv) {
  std::thread t1([]() {
    auto status =
        RunGraph("/mediapipe/mediapipe/bytetrack1.pbtxt",
                 "/mediapipe/mediapipe/examples/desktop/bytetrack/palace.mp4",
                 "/mediapipe/out_palace.mp4");
    if (!status.ok())
      LOG(ERROR) << status;
  });

  std::thread t2([]() {
    auto status = RunGraph(
        "/mediapipe/mediapipe/bytetrack3.pbtxt",
        "/mediapipe/mediapipe/examples/desktop/bytetrack/dog.mp4",
        "/mediapipe/out_dog.mp4");
    if (!status.ok())
      LOG(ERROR) << status;
  });

  while (initializedThreads < 2) {
  std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  // Signal all threads to start simultaneously
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_start = true;
  }
  g_cv.notify_all();  // ← unblocks all 3 threads at once

  t1.join();
  t2.join();
  return 0;
}