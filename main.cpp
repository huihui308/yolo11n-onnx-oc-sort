// main.cpp (Windows-compatible, ready to build)
#include <Eigen/Dense>
#include "OCsort.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <array>

// ===================== Detector config =====================
static const int    INPUT_W     = 640;
static const int    INPUT_H     = 640;
static const char*  INPUT_NAME0 = "images";
static const char*  OUT_NAME0   = "output0";
static const float  SCORE_THRES = 0.10f;
static const int    NUM_CLASSES = 80;  // YOLO11 COCO: 80 classes
static const float  NMS_IOU     = 0.10f;
static const int    DET_EVERY_N = 1;
// ===========================================================

using cv::Mat; using cv::Rect2f; using cv::Scalar; using cv::Point2f;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

static void preprocessToNCHW_fast(const cv::Mat& bgr, std::vector<float>& chw) {
    cv::Mat resized, rgb, f32;
    cv::resize(bgr, resized, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    resized.convertTo(f32, CV_32F, 1.0f/255.0f);

    if (f32.channels() > 1 && f32.size() == cv::Size(INPUT_W, INPUT_H)) {
        std::vector<cv::Mat> ch(3);
        cv::split(f32, ch);
        const int plane = INPUT_W * INPUT_H;
        chw.resize(3 * plane);
        std::memcpy(chw.data() + 0 * plane, ch[0].ptr<float>(), plane * sizeof(float));
        std::memcpy(chw.data() + 1 * plane, ch[1].ptr<float>(), plane * sizeof(float));
        std::memcpy(chw.data() + 2 * plane, ch[2].ptr<float>(), plane * sizeof(float));
    } else {
        chw.assign(f32.begin<float>(), f32.end<float>());
    }
}

static std::vector<int> nmsIndices(const std::vector<Rect2f>& boxes,
                                   const std::vector<float>& scores,
                                   float scoreThr, float iouThr) {
    std::vector<int> order; order.reserve(scores.size());
    for (int i = 0; i < (int)scores.size(); ++i)
        if (scores[i] >= scoreThr) order.push_back(i);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return scores[a] > scores[b]; });

    std::vector<int> keep;
    std::vector<bool> removed(order.size(), false);
    for (size_t i = 0; i < order.size(); ++i) {
        if (removed[i]) continue;
        keep.push_back(order[i]);
        const Rect2f a = boxes[order[i]];
        const float aArea = a.area();
        for (size_t j = i + 1; j < order.size(); ++j) {
            if (removed[j]) continue;
            const Rect2f b = boxes[order[j]];
            const float inter = (a & b).area();
            const float ovr = inter / (aArea + b.area() - inter + 1e-6f);
            if (ovr > iouThr) removed[j] = true;
        }
    }
    return keep;
}

struct LineCounter {
    cv::Point2f A, B;
    int up = 0;
    int down = 0;
    float dist_thresh = 10.f;
    int cooldown_frames = 5;
    struct TrackState { int lastSign = 0; int lastCountFrame = -999999; };
    std::unordered_map<int, TrackState> state;

    static inline float signedDistanceToLine(const Point2f& P, const Point2f& A, const Point2f& B) {
        Point2f AB = B - A;
        Point2f AP = P - A;
        float len = std::hypot(AB.x, AB.y);
        if (len < 1e-6f) return 0.f;
        float crossz = AB.x * AP.y - AB.y * AP.x;
        return crossz / len;
    }

    void update(int track_id, const cv::Point2f& center, int frame_idx) {
        float sd = signedDistanceToLine(center, A, B);
        int sign = (sd > 0.f) ? +1 : (sd < 0.f ? -1 : 0);
        auto &ts = state[track_id];
        if (ts.lastSign == 0) { ts.lastSign = sign; return; }
        if (sign != 0 && sign != ts.lastSign) {
            if (std::abs(sd) <= dist_thresh && frame_idx - ts.lastCountFrame > cooldown_frames) {
                if (ts.lastSign < 0 && sign > 0) ++up;
                else if (ts.lastSign > 0 && sign < 0) ++down;
                ts.lastCountFrame = frame_idx;
            }
        }
        ts.lastSign = sign;
    }
    void draw(cv::Mat& img) const {
        cv::line(img, A, B, Scalar(0,255,255), 2, cv::LINE_AA);
        Point2f d = B - A; float L = std::hypot(d.x, d.y);
        if (L > 1e-3f) {
            Point2f n(-d.y/L, d.x/L);
            Point2f mid = (A + B) * 0.5f;
            cv::line(img, mid - 10*n, mid + 10*n, Scalar(0,255,255), 2, cv::LINE_AA);
        }
        cv::putText(img, "Up: " + std::to_string(up) + "  Down: " + std::to_string(down),
                    {12, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,255), 2);
    }
};

int main(int argc, char *argv[]) {
    // Default paths
    const std::string default_video_path = "/home/ctfo/david/code/oc-sort/yolo11n-onnx-oc-sort/video/04150947.mp4-new.mp4";
    const std::string default_onnx_path  = "/home/ctfo/david/code/oc-sort/yolo11n-onnx-oc-sort/model/yolo11n_640.onnx";

    std::string video_path = default_video_path;
    std::string onnx_path  = default_onnx_path;

    // Parse command-line arguments
    int pos_idx = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--video" || arg == "-v") && i + 1 < argc) {
            video_path = argv[++i];
        } else if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            onnx_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [video_path] [model_path] [options]\n"
                      << "Options:\n"
                      << "  Positional args:\n"
                      << "    arg1            Video file path (default: " << default_video_path << ")\n"
                      << "    arg2            ONNX model path (default: " << default_onnx_path << ")\n"
                      << "  Named args:\n"
                      << "    --video, -v <path>   Video file path\n"
                      << "    --model, -m <path>   ONNX model path\n"
                      << "    --help, -h           Show this help message\n";
            return 0;
        } else {
            // Positional arguments: arg1 = video, arg2 = model
            if (pos_idx == 0) {
                video_path = arg;
            } else if (pos_idx == 1) {
                onnx_path = arg;
            }
            ++pos_idx;
        }
    }

    std::cout << "Using video: " << video_path << "\n";
    std::cout << "Using model: " << onnx_path << "\n";

    // OC-SORT tracker initialization
    ocsort::OCSort tracker = ocsort::OCSort(
        0, 50, 1,
        0.22136877277096445,
        1,
        "giou",
        0.3941737016672115,
        true
    );

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cout << "Error opening video file. Please check the video file path: " << video_path << std::endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 1.0 || fps > 240.0) fps = 30.0;
    cv::namedWindow("Live OC-Sort Tracker", cv::WINDOW_NORMAL);
    cv::resizeWindow("Live OC-Sort Tracker", 1280, 720);

    // ONNX Runtime session (CPU)
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_runtime");
    Ort::SessionOptions so;
    int hw = std::max(1u, std::thread::hardware_concurrency());
    so.SetIntraOpNumThreads(hw);
    so.SetInterOpNumThreads(hw);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.EnableCpuMemArena();
    so.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

    // If you have a GPU build of ONNX Runtime and want CUDA, uncomment & adapt the following:
    // OrtCUDAProviderOptions cuda{}; so.AppendExecutionProvider_CUDA(cuda);

    // Ort::Session on Linux accepts char* directly
    Ort::Session session(env, onnx_path.c_str(), so);

    const char* input_names[]  = { INPUT_NAME0 };
    const char* output_names[] = { OUT_NAME0 };

    // Persistent ONNX input tensors
    std::vector<float> chw(3 * INPUT_W * INPUT_H);
    std::array<int64_t,4> img_shape{1,3,INPUT_H,INPUT_W};
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value img_tensor = Ort::Value::CreateTensor<float>(
        memInfo, chw.data(), chw.size(), img_shape.data(), (int)img_shape.size());
    std::array<Ort::Value,1> ort_inputs = { std::move(img_tensor) };

    // Counting line
    LineCounter counter;
    counter.A = cv::Point2f(123.f, 303.f);
    counter.B = cv::Point2f(135.f, 1836.f);
    counter.dist_thresh = 15.f;
    counter.cooldown_frames = 10;

    std::cout << "OC-Sort with live video detection ready. FPS=" << fps << std::endl;

    cv::Mat frame;
    int frame_idx = 0;
    double OverAll_Time = 0;
    int stat_frames = 0;
    double stat_time = 0;
    double stat_total_time = 0;
    double stat_infer_time = 0;

    while (true) {
        auto frame_start = high_resolution_clock::now();
        if (!cap.read(frame)) {
            std::cout << "End of stream." << std::endl;
            break;
        }
        ++frame_idx;
        const int W = frame.cols;
        const int H = frame.rows;

        Eigen::MatrixXf dets;
        if ((frame_idx % DET_EVERY_N) == 0) {
            preprocessToNCHW_fast(frame, chw);

            auto infer_start = high_resolution_clock::now();
            auto outputs = session.Run(Ort::RunOptions{nullptr},
                                       input_names, ort_inputs.data(), (size_t)ort_inputs.size(),
                                       output_names, 1);
            auto infer_end = high_resolution_clock::now();
            duration<double, std::milli> infer_ms = infer_end - infer_start;
            stat_infer_time += infer_ms.count();

            // Parse YOLO output: [1, 84, 8400] = [batch, (4+80), num_predictions]
            auto* out = outputs[0].GetTensorData<float>();
            auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            int num_pred = (int)out_shape[2];  // 8400

            std::vector<Rect2f> xyxy_all;
            std::vector<float> scores_all;
            xyxy_all.reserve(num_pred);
            scores_all.reserve(num_pred);

            for (int i = 0; i < num_pred; ++i) {
                float cx = out[i];                          // channel 0
                float cy = out[num_pred + i];               // channel 1
                float bw = out[2 * num_pred + i];           // channel 2
                float bh = out[3 * num_pred + i];           // channel 3

                // Find class with highest score (channels 4 to 83)
                float class_score = 0.f;
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    float s = out[(4 + c) * num_pred + i];
                    if (s > class_score) {
                        class_score = s;
                    }
                }
                float conf = class_score;  // YOLOv8 uses class score as confidence
                if (conf < SCORE_THRES) continue;

                // YOLO output is [cx, cy, w, h] in pixels (relative to INPUT_W/INPUT_H=640)
                // Scale to actual image size
                float scale_x = (float)W / INPUT_W;
                float scale_y = (float)H / INPUT_H;

                float x1 = std::clamp(cx * scale_x - bw * scale_x * 0.5f, 0.f, (float)W - 1);
                float y1 = std::clamp(cy * scale_y - bh * scale_y * 0.5f, 0.f, (float)H - 1);
                float x2 = std::clamp(cx * scale_x + bw * scale_x * 0.5f, 0.f, (float)W - 1);
                float y2 = std::clamp(cy * scale_y + bh * scale_y * 0.5f, 0.f, (float)H - 1);

                if (x2 - x1 >= 1.f && y2 - y1 >= 1.f) {
                    xyxy_all.emplace_back(Point2f(x1, y1), Point2f(x2, y2));
                    scores_all.push_back(conf);
                }
            }

            std::vector<int> keep = nmsIndices(xyxy_all, scores_all, SCORE_THRES, NMS_IOU);

            for (int k : keep) {
                const auto& b = xyxy_all[k];
                cv::rectangle(frame,
                              cv::Rect((int)b.x, (int)b.y, (int)b.width, (int)b.height),
                              cv::Scalar(255, 0, 0), 1);
            }

            int M = (int)keep.size();
            dets.resize(M, 6);
            for (int r = 0; r < M; ++r) {
                int idx = keep[r];
                const Rect2f& b = xyxy_all[idx];
                dets(r,0) = b.x;
                dets(r,1) = b.y;
                dets(r,2) = b.x + b.width;
                dets(r,3) = b.y + b.height;
                dets(r,4) = scores_all[idx];
                dets(r,5) = 0.f; // class id
            }
        } else {
            dets.resize(0,6);
        }

        auto T_start = high_resolution_clock::now();
        std::vector<Eigen::RowVectorXf> res = tracker.update(dets);
        auto T_end = high_resolution_clock::now();
        duration<double, std::milli> ms_double = T_end - T_start;
        auto frame_end = high_resolution_clock::now();
        duration<double, std::milli> frame_ms = frame_end - frame_start;
        OverAll_Time += ms_double.count();
        stat_frames += 1;
        stat_time += ms_double.count();
        stat_total_time += frame_ms.count();
        if (stat_frames >= 30) {
            double avg_ms = stat_time / stat_frames;
            double avg_total_ms = stat_total_time / stat_frames;
            double avg_infer_ms = stat_infer_time / stat_frames;
            std::cout << "=== [Stat] Frames " << frame_idx - stat_frames + 1 << "-" << frame_idx
                      << " | Tracks: " << res.size()
                      << " | Avg YOLO: " << avg_infer_ms << " ms"
                      << " | Avg Tracker: " << avg_ms << " ms"
                      << " | Avg Total: " << avg_total_ms << " ms"
                      << " | FPS: " << (avg_total_ms > 0 ? (int)(1000.0 / avg_total_ms) : 0) << " ===" << std::endl;
            stat_frames = 0;
            stat_time = 0;
            stat_total_time = 0;
            stat_infer_time = 0;
        }

        for (const auto& j : res) {
            if (j.size() < 5) continue;
            int x1 = (int)std::round(j[0]);
            int y1 = (int)std::round(j[1]);
            int x2 = (int)std::round(j[2]);
            int y2 = (int)std::round(j[3]);
            int ID = (int)std::round(j[4]);
            float conf = (j.size() > 6) ? j[6] : 0.f;
            cv::Point2f center(0.5f*(x1 + x2), 0.5f*(y1 + y2));
            counter.update(ID, center, frame_idx);
            cv::rectangle(frame, cv::Rect(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1)),
                          cv::Scalar(3, 155, 229), 2);
            cv::putText(frame, cv::format("ID:%d %.2f", ID, conf),
                        cv::Point(x1, std::max(0, y1 - 5)), 0, 0.5,
                        cv::Scalar(229,115,115), 2, cv::LINE_AA);
        }

        counter.draw(frame);
        cv::imshow("Live OC-Sort Tracker", frame);
        int key = cv::waitKey(std::max(1, (int)std::round(1000.0 / std::max(1.0, fps))));
        if (key == 27 || key == 'q') {
            std::cout << "Program Terminate" << std::endl;
            break;
        }
    }

    double avg_cost = (frame_idx > 0) ? (OverAll_Time / frame_idx) : 0.0;
    int FPS = (avg_cost > 0.0) ? (int)(1000.0 / avg_cost) : 0;
    std::cout << "Average Time Cost: " << avg_cost << " ms  |  Avg FPS: " << FPS << std::endl;
    return 0;
}
