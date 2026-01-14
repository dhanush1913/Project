#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

enum class DetectionState {
    CLEAN,
    ADVERSARIAL,
    UNCERTAIN
};

struct DetectorConfig {
    std::vector<float> weights;  // 6 feature weights
    float threshold;              // Detection threshold
    float ood_threshold;          // Out-of-distribution threshold
};

struct DetectionResult {
    DetectionState state;
    float confidence;
    float score;
    bool requires_human_review;
};

class EmbeddedDetector {
private:
    DetectorConfig config;
    
    std::vector<float> compute_histogram(const float* image, int width, int height) {
        std::vector<float> hist(32, 0.0f);
        int total = width * height;
        
        for (int i = 0; i < total; i++) {
            int bin = static_cast<int>(image[i] * 31.999f);  // Map [0,1] to [0,31]
            bin = std::min(31, std::max(0, bin));
            hist[bin] += 1.0f;
        }
        
        for (float& h : hist) {
            h /= total;
            h = std::max(h, 1e-8f);  // Epsilon floor
        }
        
        return hist;
    }
    
    // KL divergence
    float kl_divergence(const std::vector<float>& p, const std::vector<float>& q) {
        float kl = 0.0f;
        for (size_t i = 0; i < p.size(); i++) {
            kl += p[i] * std::log(p[i] / q[i]);
        }
        return kl;
    }
    
    // JS divergence
    float js_divergence(const std::vector<float>& p, const std::vector<float>& q) {
        std::vector<float> m(p.size());
        for (size_t i = 0; i < p.size(); i++) {
            m[i] = 0.5f * (p[i] + q[i]);
        }
        
        float kl_pm = kl_divergence(p, m);
        float kl_qm = kl_divergence(q, m);
        
        return 0.5f * (kl_pm + kl_qm);
    }
    
    // Wasserstein distance (1D approximation via CDF)
    float wasserstein_1d(const std::vector<float>& p, const std::vector<float>& q) {
        std::vector<float> cdf_p(p.size());
        std::vector<float> cdf_q(q.size());
        
        // Compute CDFs
        cdf_p[0] = p[0];
        cdf_q[0] = q[0];
        for (size_t i = 1; i < p.size(); i++) {
            cdf_p[i] = cdf_p[i-1] + p[i];
            cdf_q[i] = cdf_q[i-1] + q[i];
        }
        
        // L1 distance between CDFs
        float w1 = 0.0f;
        for (size_t i = 0; i < p.size(); i++) {
            w1 += std::abs(cdf_p[i] - cdf_q[i]);
        }
        
        return w1;
    }
    
public:
    EmbeddedDetector(const DetectorConfig& cfg) : config(cfg) {}
    
    DetectionResult detect(const float* image, int width, int height,
                          const std::vector<float>& ref_histogram) {
        std::vector<float> test_hist = compute_histogram(image, width, height);
        
        float kl = kl_divergence(test_hist, ref_histogram);
        float js = js_divergence(test_hist, ref_histogram);
        float w1 = wasserstein_1d(test_hist, ref_histogram);
        
        std::vector<float> features = {kl, js, w1, 0.0f, 0.0f, 0.0f};
        
        float score = 0.0f;
        for (size_t i = 0; i < 6; i++) {
            score += config.weights[i] * features[i];
        }
        
        DetectionResult result;
        result.score = score;
        
        if (std::abs(score) > config.ood_threshold) {
            result.state = DetectionState::UNCERTAIN;
            result.confidence = 0.5f;
            result.requires_human_review = true;
        }
        else if ((std::max({kl, js, w1}) - std::min({kl, js, w1})) > 0.5f) {
            result.state = DetectionState::UNCERTAIN;
            result.confidence = 0.5f;
            result.requires_human_review = true;
        }
        else if (score > config.threshold) {
            result.state = DetectionState::ADVERSARIAL;
            result.confidence = std::min(1.0f, score / config.threshold);
            result.requires_human_review = false;
        }
        else {
            result.state = DetectionState::CLEAN;
            result.confidence = 1.0f - (score / config.threshold);
            result.requires_human_review = false;
        }
        
        return result;
    }
};


