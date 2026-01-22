#include "yolo_model.h"
#include "SortTrack.h"

// SORT算法实现

// --- 核心更新函数 ---
std::vector<TrackResponse> SortTracker::update(const std::vector<Detection> &dets)
{

    // 1. 全员预判
    for (auto &track : tracks)
    {
        track.predict();
    }

    int num_trks = (int)tracks.size();
    int num_dets = (int)dets.size();

    std::vector<bool> trk_matched(num_trks, false);
    std::vector<bool> det_matched(num_dets, false);

    // 2. 贪心匹配逻辑
    if (num_trks > 0 && num_dets > 0)
    {
        // 计算关联矩阵
        auto iou_mat = compute_iou_matrix(tracks, dets);

        while (true)
        {
            float max_iou = -1.0f;
            int b_t = -1, b_d = -1;

            for (int t = 0; t < num_trks; t++)
            {
                if (trk_matched[t])
                    continue;
                for (int d = 0; d < num_dets; d++)
                {
                    if (det_matched[d])
                        continue;
                    if (iou_mat[t][d] > max_iou)
                    {
                        max_iou = iou_mat[t][d];
                        b_t = t;
                        b_d = d;
                    }
                }
            }

            // 匹配结束条件
            if (b_t == -1 || max_iou < iou_threshold)
                break;

            trk_matched[b_t] = true;
            det_matched[b_d] = true;
            tracks[b_t].update(dets[b_d]); // 用真实观测修正卡尔曼
        }
    }

    // 3. 处理没匹配上的 Detection (新兵入伍)
    for (int d = 0; d < num_dets; d++)
    {
        if (!det_matched[d])
        {
            // emplace_back 直接构造，避免 unique_ptr 拷贝问题
            tracks.emplace_back(dets[d]);
        }
    }

    // 4. 生死判定与生成简报 (关键逻辑)
    std::vector<MicroTrack> next_frame_tracks;
    std::vector<TrackResponse> results;

    for (auto &track : tracks)
    {
        if (track.time_since_update < max_age)
        {

            // 如果连续击中次数达标，生成结果发给 GUI
            if (track.hits >= min_hits)
            {
                TrackResponse res;
                res.track_id = track.track_id;
                res.pred_det = track.pred_det; // 拷贝坐标数据
                res.hits = track.hits;
                res.age = track.age;
                results.push_back(res);
            }

            // 使用 std::move 将不准拷贝的 MicroTrack 搬运到下一帧队列
            next_frame_tracks.push_back(std::move(track));
        }
    }

    // 移交指挥权
    this->tracks = std::move(next_frame_tracks);

    return results;
}