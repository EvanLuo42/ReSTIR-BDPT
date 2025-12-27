#ifndef PBRT_CPU_RESTIRBDPT_H
#define PBRT_CPU_RESTIRBDPT_H

#include <pbrt/pbrt.h>

#include <pbrt/base/camera.h>
#include <pbrt/base/sampler.h>
#include <pbrt/bsdf.h>
#include <pbrt/cameras.h>
#include <pbrt/cpu/integrators.h>
#include <pbrt/cpu/primitive.h>
#include <pbrt/film.h>
#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace pbrt {

// Forward declarations
struct ReSTIRVertex;
struct PathSample;
struct Reservoir;

// ReSTIR Vertex types
enum class ReSTIRVertexType { Camera, Light, Surface, Medium };

// ReSTIR Path Vertex - stores path vertex information for reconnection
struct ReSTIRVertex {
    ReSTIRVertexType type = ReSTIRVertexType::Surface;
    Point3f p;    // Position
    Normal3f n;   // Geometric normal
    Normal3f ns;  // Shading normal
    Vector3f wo;  // Outgoing direction
    Float time = 0;
    SampledSpectrum beta;  // Path throughput at this vertex
    Float pdfFwd = 0;      // Forward PDF
    Float pdfRev = 0;      // Reverse PDF
    bool delta = false;    // Is delta distribution

    // For surface interactions
    SurfaceInteraction si;
    BSDF bsdf;

    // For endpoint interactions
    Light light;
    Camera camera;

    ReSTIRVertex() = default;

    bool IsConnectible() const {
        if (type == ReSTIRVertexType::Surface)
            return bsdf && IsNonSpecular(bsdf.Flags());
        if (type == ReSTIRVertexType::Light)
            return !light || !IsDeltaLight(light.Type());
        return true;  // Camera and Medium are connectible
    }

    bool IsOnSurface() const { return n != Normal3f(); }

    bool IsLight() const {
        return type == ReSTIRVertexType::Light ||
               (type == ReSTIRVertexType::Surface && si.areaLight);
    }

    // Evaluate BSDF
    SampledSpectrum f(const Vector3f &wi, TransportMode mode) const {
        if (type == ReSTIRVertexType::Surface && bsdf)
            return bsdf.f(wo, wi, mode);
        return SampledSpectrum(0.f);
    }

    // Get emission
    SampledSpectrum Le(const Vector3f &w, const SampledWavelengths &lambda) const {
        if (type == ReSTIRVertexType::Surface && si.areaLight)
            return si.areaLight.L(si.p(), si.n, si.uv, w, lambda);
        return SampledSpectrum(0.f);
    }
};

// Path sample for reservoir storage
struct PathSample {
    // Connection strategy (s, t)
    int s = 0;  // Light subpath length
    int t = 0;  // Camera subpath length

    // Reconnection vertices (y_{s-1} and z_{t-1} in paper notation)
    ReSTIRVertex lightVertex;   // y_{s-1}
    ReSTIRVertex cameraVertex;  // z_{t-1}

    // Path contribution
    SampledSpectrum contribution;
    SampledWavelengths lambda;

    // For pixel splatting (t=1 case)
    pstd::optional<Point2f> pFilm;

    // Target function value (p-hat)
    Float targetPdf = 0;

    // Original sampling PDF
    Float sourcePdf = 0;

    bool IsValid() const { return targetPdf > 0 && contribution; }

    void Reset() {
        s = t = 0;
        contribution = SampledSpectrum(0.f);
        targetPdf = 0;
        sourcePdf = 0;
        pFilm.reset();
    }
};

// Reservoir for weighted reservoir sampling
// For BDPT paths that are already unbiased estimators, we use luminance-weighted
// selection but output the path directly without additional weighting.
struct Reservoir {
    PathSample sample;  // Current selected sample
    Float wSum = 0;     // Sum of weights (for selection probability)
    int M = 0;          // Number of candidates seen
    Float W = 1;        // Output weight (1 for unbiased BDPT paths)

    void Reset() {
        sample.Reset();
        wSum = 0;
        M = 0;
        W = 1;
    }

    bool Update(const PathSample &newSample, Float weight, Float u) {
        wSum += weight;
        M++;
        if (u * wSum < weight) {
            sample = newSample;
            return true;
        }
        return false;
    }

    bool Merge(const Reservoir &other, Float targetPdf, Float u) {
        if (other.M == 0)
            return false;
        
        Float weight = targetPdf;
        wSum += weight;
        M += other.M;
        if (u * wSum < weight) {
            sample = other.sample;
            return true;
        }
        return false;
    }

    void Finalize() { 
        if (M > 0 && sample.IsValid() && sample.targetPdf > 0) {
            W = wSum / (sample.targetPdf * M);
        } else {
            W = 0.0f;
        }
    }

    void FinalizeWithMIS(Float misWeight) {
        W = (sample.IsValid()) ? misWeight : 0.0f;
    }
};

// Per-pixel reservoir buffer
class ReservoirBuffer {
  public:
    ReservoirBuffer() = default;

    void Initialize(Point2i resolution) {
        this->resolution = resolution;
        reservoirs.resize(resolution.x * resolution.y);
        Reset();
    }

    void Reset() {
        for (auto &r : reservoirs)
            r.Reset();
    }

    Reservoir &operator()(Point2i p) { return reservoirs[p.y * resolution.x + p.x]; }

    const Reservoir &operator()(Point2i p) const {
        return reservoirs[p.y * resolution.x + p.x];
    }

    bool InBounds(Point2i p) const {
        return p.x >= 0 && p.x < resolution.x && p.y >= 0 && p.y < resolution.y;
    }

    Point2i Resolution() const { return resolution; }

  private:
    Point2i resolution;
    std::vector<Reservoir> reservoirs;
};

// ReSTIR BDPT Integrator
class ReSTIRBDPTIntegrator : public ImageTileIntegrator {
  public:
    // Constructor
    ReSTIRBDPTIntegrator(Camera camera, Sampler sampler, Primitive aggregate,
                         std::vector<Light> lights, int maxDepth, int spatialIterations,
                         int spatialNeighbors, Float spatialRadius, bool temporalReuse,
                         bool regularize);

    // Factory method
    static std::unique_ptr<ReSTIRBDPTIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const override;

    // Main render loop (overrides ImageTileIntegrator::Render)
    void Render() override;

    // Per-pixel evaluation (required by ImageTileIntegrator)
    void EvaluatePixelSample(Point2i pPixel, int sampleIndex, Sampler sampler,
                             ScratchBuffer &scratchBuffer) override;

  private:
    // Generate initial path samples for a pixel
    void GenerateInitialSamples(Point2i pPixel, int sampleIndex, Sampler sampler,
                                ScratchBuffer &scratchBuffer,
                                std::vector<PathSample> &samples);

    // Build camera subpath
    int GenerateCameraSubpath(const RayDifferential &ray, SampledWavelengths &lambda,
                              Sampler sampler, ScratchBuffer &scratchBuffer, int maxDepth,
                              std::vector<ReSTIRVertex> &path);

    // Build light subpath
    int GenerateLightSubpath(SampledWavelengths &lambda, Sampler sampler,
                             ScratchBuffer &scratchBuffer, int maxDepth, Float time,
                             std::vector<ReSTIRVertex> &path);

    // Connect subpaths and compute contribution
    SampledSpectrum ConnectSubpaths(const std::vector<ReSTIRVertex> &lightVertices,
                                    const std::vector<ReSTIRVertex> &cameraVertices,
                                    int s, int t, SampledWavelengths &lambda,
                                    Sampler sampler, pstd::optional<Point2f> *pFilm,
                                    Float *misWeight);

    // Compute geometry term
    SampledSpectrum G(const ReSTIRVertex &v0, const ReSTIRVertex &v1,
                      const SampledWavelengths &lambda) const;

    // Compute MIS weight for connection strategy
    Float MISWeight(const std::vector<ReSTIRVertex> &lightVertices,
                    const std::vector<ReSTIRVertex> &cameraVertices, int s, int t) const;

    // Target PDF for ReSTIR (luminance-based)
    Float TargetPdf(const PathSample &sample) const;

    // Spatial resampling pass
    void SpatialResample(Point2i pPixel, Sampler sampler, ReservoirBuffer &current,
                         const ReservoirBuffer &source);

    // Temporal resampling pass
    void TemporalResample(Point2i pPixel, Sampler sampler, ReservoirBuffer &current,
                          const ReservoirBuffer &previous);

    // Check visibility for reconnection
    bool CheckVisibility(const ReSTIRVertex &v0, const ReSTIRVertex &v1) const;

    // Shift mapping for sample reuse
    bool ShiftSample(const PathSample &source, Point2i targetPixel, PathSample &shifted,
                     Float *jacobian) const;

    // Member variables
    int maxDepth;
    int spatialIterations;
    int spatialNeighbors;
    Float spatialRadius;
    bool temporalReuse;
    bool regularize;

    LightSampler lightSampler;

    // Reservoir buffers (double-buffered for temporal reuse)
    ReservoirBuffer reservoirBuffer[2];
    int currentBuffer = 0;

    // Accumulation buffer for final output
    struct PixelAccum {
        SampledSpectrum L;
        SampledWavelengths lambda;
        Float weight = 0;
    };
    std::vector<PixelAccum> accumBuffer;

    // Thread-local storage for subpaths
    mutable ThreadLocal<std::vector<ReSTIRVertex>> tlCameraVertices;
    mutable ThreadLocal<std::vector<ReSTIRVertex>> tlLightVertices;
};

}  // namespace pbrt

#endif  // PBRT_CPU_RESTIRBDPT_H
